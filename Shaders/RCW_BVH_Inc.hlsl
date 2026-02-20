// bvh基础叶节点：一条边 (已废弃，保留用于参考)
struct EdgeBVH
{
    float2 start;
    float2 end;
    int matIdx;
};
// BVH 节点结构 (已废弃，保留用于参考)
struct LBVHNode
{
    float2 Min;       // AABB 最小值
    float2 Max;       // AABB 最大值
    int LeftChild;     // 左孩子索引
    int RightChild;    // 右孩子索引
    int Parent;        // 父节点索引
    int ObjectIndex;   //如果是叶子，这里存储原始边的索引(index数组的内容)；如果是内部节点，为-1
};

// 压缩后的节点-边数据 (当前使用)
struct LBVHNodeGpu
{
    // 复用区域 1: 几何/空间信息 (8 bytes)
    // 内部BVH节点: AABB Min xy
    // 叶子BVH节点: Edge Start xy
    float2 PosA;
    // 复用区域 2: 几何/空间信息 (8 bytes)
    // 内部BVH节点: AABB Max (xy)
    // 叶子BVH节点: Edge End (xy)
    float2 PosB;
    // 复用区域 3: 索引/元数据 (4 bytes)
    // 内部BVH节点: Left Child Index (>= 0) 左子的索引
    // 叶子BVH节点: Bitwise NOT of Material Index (< 0) -> ~MatIdx 全部取反，作为材质的索引
    int IndexData;
    // 复用区域 4: 辅助索引 (4 bytes)
    // 内部BVH节点: Right Child Index  右子的索引
    // 叶子BVH节点: Unused (or Padding) 暂无用
    int RightChild;
};

// 材质数据
struct MaterialData
{
    float4 BasicColor;
    float4 Emission;
    float4 uvBox;
    
    float Density;
    int TextureIndex;
    
    // 对应 C# 的占位符，保持总大小一致
    float2 _padding; 
};

// StructuredBuffer <EdgeBVH>  _BVH_Edge_Buffer;      // 已废弃，合并到 NodeEdge
// StructuredBuffer <LBVHNode> _BVH_Node_Buffer;      // 已废弃，合并到 NodeEdge
StructuredBuffer <LBVHNodeGpu> _BVH_NodeEdge_Buffer;  // 紧凑格式：内部节点+叶子边数据
StructuredBuffer <MaterialData> _BVH_Material_Buffer;
int _BVH_Root_Index;

// 世界空间射线
struct RayWS
{
    float2 Origin;
    float2 Direction;
};

struct IntersectsRaySegmentResult
{
    float2 hitPoint;
    float2 hitNormal;
    int nodeIndex;      // 命中的叶子节点索引
};

// =========================================================
// 辅助函数：从叶子节点获取材质索引
// =========================================================
int GetMaterialIndex(in LBVHNodeGpu leafNode)
{
    // 叶子节点的 IndexData = ~matIdx，还原需要再次取反
    return ~leafNode.IndexData;
}

// =========================================================
// 求交：射线和AABB包围盒求交 (使用紧凑格式节点)
// 对于内部节点: PosA = AABB Min, PosB = AABB Max
// =========================================================
bool IntersectsRayAABB(RayWS ray, in LBVHNodeGpu node)
{
    // 1. 处理射线方向，防止除以 0
    float2 dir = ray.Direction;
    if (abs(dir.x) < 1e-9) dir.x = 1e-9 * (dir.x >= 0 ? 1.0 : -1.0);
    if (abs(dir.y) < 1e-9) dir.y = 1e-9 * (dir.y >= 0 ? 1.0 : -1.0);
    
    float2 invDir = 1.0f / dir;

    // 2. 计算射线与 AABB 各个平面的相交时间 t
    // 内部节点: PosA = Min, PosB = Max
    float2 t0 = (node.PosA - ray.Origin) * invDir;
    float2 t1 = (node.PosB - ray.Origin) * invDir;

    // 3. 确定进入时间(tMin)和离开时间(tMax)
    float2 tMinVec = min(t0, t1);
    float2 tMaxVec = max(t0, t1);

    // 4. 计算最终的进入和离开时间
    float tEnter = max(tMinVec.x, tMinVec.y);
    float tExit  = min(tMaxVec.x, tMaxVec.y);

    // 5. 判定相交
    return (tExit >= tEnter) && (tExit >= 0.0f);
}


// =========================================================
// 求交：射线与线段求交 (使用紧凑格式叶子节点)
// 对于叶子节点: PosA = Edge Start, PosB = Edge End
// =========================================================
bool IntersectsRaySegment(RayWS ray, in LBVHNodeGpu leafNode, out IntersectsRaySegmentResult result)
{
    // 1. 初始化
    result.hitPoint = float2(0, 0);
    result.hitNormal = float2(0, 0);
    result.nodeIndex = -1;

    // 2. 准备向量
    // 叶子节点: PosA = edge.start, PosB = edge.end
    float2 p = ray.Origin;
    float2 r = ray.Direction;
    float2 q = leafNode.PosA;                    // edge.start
    float2 s = leafNode.PosB - leafNode.PosA;    // edge vector

    // 3. 计算分母 (r x s)
    float rCrossS = r.x * s.y - r.y * s.x;

    // 4. 平行检测
    if (abs(rCrossS) < 1e-7)
    {
        return false;
    }

    // 5. 计算 t (射线参数) 和 u (线段参数)
    float2 qp = q - p;
    float qpCrossS = qp.x * s.y - qp.y * s.x;
    float qpCrossR = qp.x * r.y - qp.y * r.x;

    float t = qpCrossS / rCrossS;
    float u = qpCrossR / rCrossS;

    // 6. 判定相交
    if (t > 0.0f && u >= 0.0f && u <= 1.0f)
    {
        result.hitPoint = p + r * t;

        // 7. 计算法线
        float2 normalVec = float2(s.y, -s.x);
        result.hitNormal = normalize(normalVec);
        
        return true;
    }

    return false;
}


#define MAX_RECUR_DEEP 32
// =========================================================
// 手动递归：射线与BVH求交 (使用紧凑格式 _BVH_NodeEdge_Buffer)
// =========================================================
bool IntersectRayBVH(RayWS ray, out IntersectsRaySegmentResult result)
{
    // 1. 初始化结果
    result.hitPoint = float2(0, 0);
    result.hitNormal = float2(0, 0);
    result.nodeIndex = -1;
    
    if (_BVH_Root_Index == -1)
        return false;

    // 2. 初始化遍历状态
    float closestDist = 1e30;
    bool hitFound = false;

    // 3. 准备栈 (模拟递归)
    int nodeStack[MAX_RECUR_DEEP]; 
    int stackTop = 0;
    nodeStack[0] = _BVH_Root_Index;

    // 4. 开始遍历 Loop
    [loop]
    while (stackTop >= 0)
    {
        int nodeIdx = nodeStack[stackTop];
        stackTop--;

        if (nodeIdx == -1) continue;

        // 获取紧凑格式节点数据
        LBVHNodeGpu node = _BVH_NodeEdge_Buffer[nodeIdx];

        // 判断是否为叶子节点: IndexData < 0 表示叶子
        bool isLeaf = (node.IndexData < 0);

        if (isLeaf)
        {
            // === 叶子节点处理 ===
            // 叶子节点: PosA = edge.start, PosB = edge.end
            // IndexData = ~matIdx (按位取反)
            IntersectsRaySegmentResult tempResult;

            if (IntersectsRaySegment(ray, node, tempResult))
            {
                float dist = distance(ray.Origin, tempResult.hitPoint);

                if (dist < closestDist)
                {
                    closestDist = dist;
                    result = tempResult;
                    result.nodeIndex = nodeIdx; // 记录叶子节点索引
                    hitFound = true;
                }
            }
        }
        else
        {
            // === 内部节点处理 ===
            // 内部节点: PosA = AABB Min, PosB = AABB Max
            // IndexData = LeftChild, RightChild = RightChild

            // AABB 剔除测试
            if (!IntersectsRayAABB(ray, node))
            {
                continue;
            }

            // 将左右孩子压入栈中
            if (stackTop < MAX_RECUR_DEEP - 2) 
            {
                stackTop++;
                nodeStack[stackTop] = node.IndexData;   // LeftChild
                
                stackTop++;
                nodeStack[stackTop] = node.RightChild;
            }
        }
    }

    return hitFound;
}

#define MAX_INTERSECTS 4
struct IntersectsRaySegmentResultArray
{
    int intersectsCount;
    IntersectsRaySegmentResult results[MAX_INTERSECTS];
};

// =========================================================
// 核心实现：带最大距离限制的 BVH 射线求交（收集多个交点）
// 使用紧凑格式 _BVH_NodeEdge_Buffer
// =========================================================
bool IntersectRayBVHArray(RayWS ray, float maxDistance, out IntersectsRaySegmentResultArray result)
{
    // 1. 初始化结果
    result.intersectsCount = 0;
    
    [unroll]
    for (int i = 0; i < MAX_INTERSECTS; i++)
    {
        result.results[i].hitPoint = float2(0, 0);
        result.results[i].hitNormal = float2(0, 0);
        result.results[i].nodeIndex = -1;
    }

    // 辅助数组：用于存储距离
    float hitDistances[MAX_INTERSECTS]; 
    [unroll]
    for(int j = 0; j < MAX_INTERSECTS; j++) hitDistances[j] = 1e30;

    if (_BVH_Root_Index == -1)
        return false;

    // 栈
    int nodeStack[MAX_RECUR_DEEP];
    int stackTop = 0;
    nodeStack[0] = _BVH_Root_Index;

    // 开始遍历
    [loop]
    while (stackTop >= 0)
    {
        int nodeIdx = nodeStack[stackTop];
        stackTop--;

        if (nodeIdx == -1) continue;

        // 获取紧凑格式节点数据
        LBVHNodeGpu node = _BVH_NodeEdge_Buffer[nodeIdx];

        // 判断是否为叶子节点: IndexData < 0 表示叶子
        bool isLeaf = (node.IndexData < 0);

        if (isLeaf)
        {
            // === 叶子节点处理 ===
            IntersectsRaySegmentResult tempResult;

            if (IntersectsRaySegment(ray, node, tempResult))
            {
                float dist = distance(ray.Origin, tempResult.hitPoint);

                // 剔除逻辑
                if (dist > maxDistance) continue;
                if (result.intersectsCount == MAX_INTERSECTS && dist >= hitDistances[MAX_INTERSECTS - 1])
                {
                    continue;
                }

                // GPU 友好的插入排序
                int insertPos = result.intersectsCount;
                
                [unroll]
                for (int k = 0; k < MAX_INTERSECTS; k++)
                {
                    if (dist < hitDistances[k])
                    {
                        insertPos = min(insertPos, k);
                    }
                }

                if (insertPos < MAX_INTERSECTS)
                {
                    // 元素后移 (Shift)
                    [unroll]
                    for (int m = MAX_INTERSECTS - 1; m > 0; m--)
                    {
                        if (m > insertPos)
                        {
                            result.results[m] = result.results[m-1];
                            hitDistances[m] = hitDistances[m-1];
                        }
                    }

                    // 填入新数据
                    [unroll]
                    for (int n = 0; n < MAX_INTERSECTS; n++)
                    {
                        if (n == insertPos)
                        {
                            result.results[n] = tempResult;
                            result.results[n].nodeIndex = nodeIdx; // 记录叶子节点索引
                            hitDistances[n] = dist;
                        }
                    }

                    // 更新计数
                    if (result.intersectsCount < MAX_INTERSECTS)
                        result.intersectsCount++;
                }
            }
        }
        else
        {
            // === 内部节点处理 ===
            // AABB 剔除测试
            if (!IntersectsRayAABB(ray, node))
            {
                continue;
            }

            // 将左右孩子压入栈中
            if (stackTop < MAX_RECUR_DEEP - 2)
            {
                stackTop++;
                nodeStack[stackTop] = node.IndexData;   // LeftChild
                stackTop++;
                nodeStack[stackTop] = node.RightChild;
            }
        }
    }

    return (result.intersectsCount > 0);
}

// 重载：无距离限制版本
bool IntersectRayBVHArray(RayWS ray, out IntersectsRaySegmentResultArray result)
{
    return IntersectRayBVHArray(ray, 1e30, result);
}

#define MAX_RAYMARCHING_INTERVALS 4
struct RayMarchingInterval
{
    int matIdx;
    float2 start;
    float2 end;
};
