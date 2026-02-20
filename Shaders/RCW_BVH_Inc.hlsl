// bvh基础叶节点：一条边
struct EdgeBVH
{
    float2 start;
    float2 end;
    int matIdx;
};
// BVH 节点结构
struct LBVHNode
{
    float2 Min;       // AABB 最小值
    float2 Max;       // AABB 最大值
    int LeftChild;     // 左孩子索引
    int RightChild;    // 右孩子索引
    int Parent;        // 父节点索引
    int ObjectIndex;   //如果是叶子，这里存储原始边的索引(index数组的内容)；如果是内部节点，为-1
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

StructuredBuffer <EdgeBVH>  _BVH_Edge_Buffer;
StructuredBuffer <LBVHNode> _BVH_Node_Buffer;
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
    int nodeIndex;
};

// 求交：射线和AABB包围盒求交
bool IntersectsRayAABB(RayWS ray, in LBVHNode node)
{
    // 1. 处理射线方向，防止除以 0
    // 如果方向分量极小，将其设为一个非零的极小值，保持符号，以免 invDir 变成 NaN 或 Inf 导致逻辑错误
    float2 dir = ray.Direction;
    if (abs(dir.x) < 1e-9) dir.x = 1e-9 * (dir.x >= 0 ? 1.0 : -1.0);
    if (abs(dir.y) < 1e-9) dir.y = 1e-9 * (dir.y >= 0 ? 1.0 : -1.0);
    
    float2 invDir = 1.0f / dir;

    // 2. 计算射线与 AABB 各个平面的相交时间 t
    // 公式: t = (Plane - Origin) / Direction
    float2 t0 = (node.Min - ray.Origin) * invDir;
    float2 t1 = (node.Max - ray.Origin) * invDir;

    // 3. 确定进入时间(tMin)和离开时间(tMax)
    // min() 得到的是射线进入 Slab 的时间，max() 得到的是离开 Slab 的时间
    float2 tMinVec = min(t0, t1);
    float2 tMaxVec = max(t0, t1);

    // 4. 计算最终的进入和离开时间
    // 射线必须进入所有维度的 Slab 才算进入盒子，所以取 max(tMin...)
    // 射线只要离开任意维度的 Slab 就算离开盒子，所以取 min(tMax...)
    float tEnter = max(tMinVec.x, tMinVec.y);
    float tExit  = min(tMaxVec.x, tMaxVec.y);

    // 5. 判定相交
    // 条件 A (tExit >= tEnter): 射线在两个轴上的重叠区间有效（没有错开）
    // 条件 B (tExit >= 0): 盒子在射线前方（或者射线起点在盒子内部）
    return (tExit >= tEnter) && (tExit >= 0.0f);
}


// 求交：射线与线段求交
bool IntersectsRaySegment(RayWS ray, in EdgeBVH edge, out IntersectsRaySegmentResult result)
{
    // 1. 初始化
    result.hitPoint = float2(0, 0);
    result.hitNormal = float2(0, 0);
    result.nodeIndex = -1; // 具体的索引需要在外部调用时赋值

    // 2. 准备向量
    float2 p = ray.Origin;
    float2 r = ray.Direction;
    float2 q = edge.start;
    float2 s = edge.end - edge.start; // 线段向量 (Edge Vector)

    // 3. 计算分母 (r x s)
    // 2D 叉乘: a.x * b.y - a.y * b.x
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
    // t > 0: 射线前方
    // 0 <= u <= 1: 线段内部
    if (t > 0.0f && u >= 0.0f && u <= 1.0f)
    {
        result.hitPoint = p + r * t;

        // 7. 计算法线 (核心修改部分)
        // 题目要求：法线 cross 有向边 > 0
        // 设边为 (x, y)，则法线为 (y, -x)
        // 验证：(y, -x) cross (x, y) = y*y - (-x)*x = y^2 + x^2 > 0
        // 几何意义：对于顺时针(CW)多边形，这通常指向多边形外部（右手边）
        float2 normalVec = float2(s.y, -s.x);
        
        result.hitNormal = normalize(normalVec);
        
        return true;
    }

    return false;
}


#define MAX_RECUR_DEEP 32
// 手动递归：射线与BVH求交
bool IntersectRayBVH(RayWS ray, out IntersectsRaySegmentResult result)
{
    // 1. 初始化结果
    result.hitPoint = float2(0, 0);
    result.hitNormal = float2(0, 0);
    result.nodeIndex = -1;
    
    // 如果没有根节点，直接返回
    if (_BVH_Root_Index == -1)
        return false;

    // 2. 初始化遍历状态
    float closestDist = 1e30; // 初始化为无穷大
    bool hitFound = false;

    // 3. 准备栈 (模拟递归)
    // 深度通常取决于树的高度，32-64 对于大多数 2D 场景足够
    int nodeStack[MAX_RECUR_DEEP]; 
    int stackTop = 0;
    
    // 将根节点压栈
    nodeStack[0] = _BVH_Root_Index;

    // 4. 开始遍历 Loop
    [loop]
    while (stackTop >= 0)
    {
        // 4.1 弹出当前节点索引
        int nodeIdx = nodeStack[stackTop];
        stackTop--;

        // 容错检查
        if (nodeIdx == -1) continue;

        // 获取节点数据
        LBVHNode node = _BVH_Node_Buffer[nodeIdx];

        // 4.2 AABB 剔除测试
        // 如果射线没有击中这个节点的包围盒，那么它的所有子节点都不可能被击中 -> 剪枝
        if (!IntersectsRayAABB(ray, node))
        {
            continue;
        }

        // 4.3 判断是否为叶子节点 (ObjectIndex != -1)
        if (node.ObjectIndex != -1)
        {
            // --- 叶子节点处理 ---
            EdgeBVH edge = _BVH_Edge_Buffer[node.ObjectIndex];
            IntersectsRaySegmentResult tempResult;

            // 进行精确的射线-线段求交
            if (IntersectsRaySegment(ray, edge, tempResult))
            {
                // 计算距离
                float dist = distance(ray.Origin, tempResult.hitPoint);

                // 如果发现了更近的交点，更新结果
                if (dist < closestDist)
                {
                    closestDist = dist;
                    result = tempResult;
                    result.nodeIndex = node.ObjectIndex; // 记录原始边的索引
                    hitFound = true;
                }
            }
        }
        else
        {
            // --- 内部节点处理 ---
            // 将左右孩子压入栈中，等待后续处理
            // 注意：防止栈溢出
            if (stackTop < MAX_RECUR_DEEP - 2) 
            {
                // 简单的压栈顺序。
                // 进阶优化：可以先计算射线到左右孩子AABB的距离，
                // 先压入“远”的孩子，后压入“近”的孩子，这样循环会先处理“近”的孩子，
                // 有助于更快收缩 closestDist 从而进行更多剪枝。
                // 但在这里我们使用基础逻辑：
                
                stackTop++;
                nodeStack[stackTop] = node.LeftChild;
                
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
// =========================================================
bool IntersectRayBVHArray(RayWS ray, float maxDistance, out IntersectsRaySegmentResultArray result)
{
    // 1. 初始化结果
    result.intersectsCount = 0;
    
    // 【关键修改】使用 [unroll] 确保初始化循环展开
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

    // 4. 开始遍历 (外层保持 loop，处理不确定的树深)
    [loop]
    while (stackTop >= 0)
    {
        int nodeIdx = nodeStack[stackTop];
        stackTop--;

        if (nodeIdx == -1) continue;

        // 获取节点数据
        LBVHNode node = _BVH_Node_Buffer[nodeIdx];

        if (!IntersectsRayAABB(ray, node))
        {
            continue;
        }

        if (node.ObjectIndex != -1)
        {
            EdgeBVH edge = _BVH_Edge_Buffer[node.ObjectIndex];
            IntersectsRaySegmentResult tempResult;

            if (IntersectsRaySegment(ray, edge, tempResult))
            {
                float dist = distance(ray.Origin, tempResult.hitPoint);

                // 剔除逻辑
                if (dist > maxDistance) continue;
                if (result.intersectsCount == MAX_INTERSECTS && dist >= hitDistances[MAX_INTERSECTS - 1])
                {
                    continue;
                }

                // =================================================
                // 【核心修复】GPU 友好的插入排序
                // 消除所有对局部数组的动态索引访问 (Variable Indexing)
                // =================================================

                // 1. 寻找插入位置
                int insertPos = result.intersectsCount;
                
                // 使用 [unroll] 强制展开，边界使用常量 MAX_INTERSECTS
                [unroll]
                for (int k = 0; k < MAX_INTERSECTS; k++)
                {
                    // 只要发现比当前距离大的位置，就尝试更新 insertPos
                    // 因为数组是有序的，第一个满足条件的就是正确位置，min 保证了锁定第一个
                    if (dist < hitDistances[k])
                    {
                        insertPos = min(insertPos, k);
                    }
                }

                if (insertPos < MAX_INTERSECTS)
                {
                    // 2. 元素后移 (Shift)
                    // 从后向前遍历，同样使用常量边界 + 条件赋值
                    [unroll]
                    for (int m = MAX_INTERSECTS - 1; m > 0; m--)
                    {
                        // 只有在插入点之后的元素才需要移动
                        // 编译器会将其转化为条件传送指令 (movc)，避免动态索引
                        if (m > insertPos)
                        {
                            result.results[m] = result.results[m-1];
                            hitDistances[m] = hitDistances[m-1];
                        }
                    }

                    // 3. 填入新数据
                    // 避免使用 result.results[insertPos] = ... 这种动态写入
                    // 而是遍历所有槽位，只写入匹配的那个
                    [unroll]
                    for (int n = 0; n < MAX_INTERSECTS; n++)
                    {
                        if (n == insertPos)
                        {
                            result.results[n] = tempResult;
                            result.results[n].nodeIndex = node.ObjectIndex;
                            hitDistances[n] = dist;
                        }
                    }

                    // 4. 更新计数
                    if (result.intersectsCount < MAX_INTERSECTS)
                        result.intersectsCount++;
                }
            }
        }
        else
        {
            if (stackTop < MAX_RECUR_DEEP - 2)
            {
                stackTop++;
                nodeStack[stackTop] = node.LeftChild;
                stackTop++;
                nodeStack[stackTop] = node.RightChild;
            }
        }
    }

    return (result.intersectsCount > 0);
}

// 重载保持不变
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
