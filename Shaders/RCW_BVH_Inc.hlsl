#ifndef RCW_BVH_INC_HLSL
#define RCW_BVH_INC_HLSL

// =========================================================
// 半透明物体宏开关
// 启用时：使用 IntersectRayBVHArray 收集多个交点，计算介质距离-指数衰减
// 禁用时：使用 IntersectRayBVH 只取最近交点，所有物体视为完全不透明
// =========================================================
//#define ENABLE_TRANSLUCENT_OBJECTS

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
    float4 uvMatrix;
    float2 uvTranslation;
    
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
    int matIdx;
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
bool IntersectsRayAABB(RayWS ray, float2 invDir, in LBVHNodeGpu node)
{
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
bool IntersectsRaySegment(RayWS ray, in LBVHNodeGpu leafNode, int matIdx, out IntersectsRaySegmentResult result)
{
    // 1. 初始化
    result.hitPoint = float2(0, 0);
    result.hitNormal = float2(0, 0);
    result.nodeIndex = -1;
    result.matIdx = matIdx;

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

    // 预计算invDir
    float2 dir = ray.Direction;
    if (abs(dir.x) < 1e-9) dir.x = 1e-9 * (dir.x >= 0 ? 1.0 : -1.0);
    if (abs(dir.y) < 1e-9) dir.y = 1e-9 * (dir.y >= 0 ? 1.0 : -1.0);
    float2 invDir = 1.0f / dir;

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

            int matIdx = ~node.IndexData;
            
            if (IntersectsRaySegment(ray, node, matIdx, tempResult))
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
            if (!IntersectsRayAABB(ray, invDir, node))
            {
                continue;
            }

            // 将左右孩子压入栈中
            if (stackTop < MAX_RECUR_DEEP - 2) 
            {
                stackTop++;
                nodeStack[stackTop] = node.RightChild;
                stackTop++;
                nodeStack[stackTop] = node.IndexData;   // LeftChild
            }
        }
    }

    return hitFound;
}

// =========================================================
// 带最大距离限制的 BVH 射线求交（只返回最近交点）
// 用于不透明物体模式下的简化阴影计算
// =========================================================
bool IntersectRayBVH(RayWS ray, float maxDistance, out IntersectsRaySegmentResult result)
{
    // 1. 初始化结果
    result.hitPoint = float2(0, 0);
    result.hitNormal = float2(0, 0);
    result.nodeIndex = -1;
    result.matIdx = -1;
    
    if (_BVH_Root_Index == -1)
        return false;

    // 2. 初始化遍历状态
    float closestDist = maxDistance;
    bool hitFound = false;

    // 3. 准备栈 (模拟递归)
    int nodeStack[MAX_RECUR_DEEP]; 
    int stackTop = 0;
    nodeStack[0] = _BVH_Root_Index;

    // 预计算invDir
    float2 dir = ray.Direction;
    if (abs(dir.x) < 1e-9) dir.x = 1e-9 * (dir.x >= 0 ? 1.0 : -1.0);
    if (abs(dir.y) < 1e-9) dir.y = 1e-9 * (dir.y >= 0 ? 1.0 : -1.0);
    float2 invDir = 1.0f / dir;

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
            IntersectsRaySegmentResult tempResult;
            int matIdx = ~node.IndexData;
            
            if (IntersectsRaySegment(ray, node, matIdx, tempResult))
            {
                float dist = distance(ray.Origin, tempResult.hitPoint);

                // 只考虑在 maxDistance 范围内的交点
                if (dist > 0.01 && dist < closestDist)
                {
                    closestDist = dist;
                    result = tempResult;
                    result.nodeIndex = nodeIdx;
                    hitFound = true;
                }
            }
        }
        else
        {
            // === 内部节点处理 ===
            if (!IntersectsRayAABB(ray, invDir, node))
            {
                continue;
            }

            // 将左右孩子压入栈中
            if (stackTop < MAX_RECUR_DEEP - 2) 
            {
                stackTop++;
                nodeStack[stackTop] = node.RightChild;
                stackTop++;
                nodeStack[stackTop] = node.IndexData;   // LeftChild
            }
        }
    }

    return hitFound;
}

#define MAX_INTERSECTS 3
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

    // 预计算invDir
    float2 dir = ray.Direction;
    if (abs(dir.x) < 1e-9) dir.x = 1e-9 * (dir.x >= 0 ? 1.0 : -1.0);
    if (abs(dir.y) < 1e-9) dir.y = 1e-9 * (dir.y >= 0 ? 1.0 : -1.0);
    float2 invDir = 1.0f / dir;

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

            int matIdx = ~node.IndexData; 

            if (IntersectsRaySegment(ray, node, matIdx, tempResult))
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
            if (!IntersectsRayAABB(ray, invDir, node))
            {
                continue;
            }

            // 将左右孩子压入栈中
            if (stackTop < MAX_RECUR_DEEP - 2)
            {
                stackTop++;
                nodeStack[stackTop] = node.RightChild;
                stackTop++;
                nodeStack[stackTop] = node.IndexData;   // LeftChild
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


void GetIntervals(RayWS ray, IntersectsRaySegmentResultArray results, float maxLightLength, out RayMarchingInterval intervals[MAX_RAYMARCHING_INTERVALS], out int intervalCount)
{
    // 初始化的interval数据
    float2 currStart = ray.Origin;
    float2 currEnd = ray.Origin + ray.Direction * maxLightLength;
    int currMatIdx = results.results[0].matIdx;//GetMaterialIndex(_BVH_NodeEdge_Buffer[results.results[0].nodeIndex]);

    intervalCount = 0;
    bool endflag = true;
    for (int j = 0; j < results.intersectsCount; ++j)
    {
        IntersectsRaySegmentResult currResult = results.results[j];
                
        // 当前点是出点，
        if (dot(currResult.hitNormal, ray.Direction) > 0.0)
        {
            currEnd = currResult.hitPoint;
                    
            // 和入点一起组成一个区间
            intervals[intervalCount].start = currStart;
            intervals[intervalCount].end = currEnd;
            intervals[intervalCount].matIdx = currMatIdx;
            endflag = true;
            intervalCount++;
        }
        // 否则是入点
        else
        {
            currStart = currResult.hitPoint;
            currMatIdx = currResult.matIdx;//GetMaterialIndex(_BVH_NodeEdge_Buffer[currResult.nodeIndex]);
            endflag = false;
        }
    }
    // 特判：如果循环结束了endflag却是false，那么说明区间开放了
    if (endflag == false)
    {
        currEnd = ray.Origin + ray.Direction * maxLightLength;
        intervals[intervalCount].start = currStart;
        intervals[intervalCount].end = currEnd;
        intervals[intervalCount].matIdx = currMatIdx;
        endflag = true;
        intervalCount++;
    }
}



/// <summary>
/// 使用计算好的变换，将世界空间点转换为Atlas UV坐标（用于调试验证）
/// </summary>
float2 WorldToAtlasUV(in float2 worldPoint, in float4 uvMatrix, in float2 uvTranslation)
{
    return float2(
        uvMatrix.x * worldPoint.x + uvMatrix.y * worldPoint.y + uvTranslation.x,
        uvMatrix.z * worldPoint.x + uvMatrix.w * worldPoint.y + uvTranslation.y
    );
}




// =========================================================
// 光照模型函数（便于统一修改为半兰伯特或其他 BRDF）
// =========================================================

/// <summary>
/// 标准兰伯特光照
/// </summary>
/// <param name="normalWS">世界空间法线（归一化）</param>
/// <param name="lightDirWS">从片元指向光源的方向（归一化）</param>
/// <returns>光照强度 [0, 1]</returns>
float LambertLighting(float3 normalWS, float3 lightDirWS)
{
    return saturate(dot(normalWS, lightDirWS));
}

/// <summary>
/// 半兰伯特光照（可选，用于更柔和的阴影过渡）
/// </summary>
float HalfLambertLighting(float3 normalWS, float3 lightDirWS)
{
    float NdotL = dot(normalWS, lightDirWS);
    return NdotL * 0.5 + 0.5;
}

/// <summary>
/// 可配置的兰伯特光照
/// mode: 0 = 标准兰伯特, 1 = 半兰伯特
/// </summary>
float ConfigurableLambert(float3 normalWS, float3 lightDirWS, int mode)
{
    if (mode == 1)
        return HalfLambertLighting(normalWS, lightDirWS);
    else
        return LambertLighting(normalWS, lightDirWS);
}

// 当前使用的光照模式（可通过 Shader.SetGlobalInt 设置）
// 0 = 标准兰伯特, 1 = 半兰伯特
int _RCWB_LightingMode;

/// <summary>
/// 全局光照计算入口（使用全局配置的光照模式）
/// </summary>
float CalculateLighting(float3 normalWS, float3 lightDirWS)
{
    return ConfigurableLambert(normalWS, lightDirWS, _RCWB_LightingMode);
}

// =========================================================
// BVH 阴影/遮挡计算
// =========================================================

/// <summary>
/// 计算从片元到光源的阴影衰减
/// 启用半透明时：使用 GetIntervals 构建介质区间，计算 exp(-distance * density) 透射率
/// 禁用半透明时：只检查是否有任何遮挡，有则返回 0
/// </summary>
/// <param name="worldPos">片元世界位置 (2D)</param>
/// <param name="lightPos">光源世界位置 (2D)</param>
/// <returns>阴影系数 [0, 1]，0 = 完全遮挡，1 = 无遮挡</returns>
float CalculateShadowAttenuation(float2 worldPos, float2 lightPos)
{
    float2 toLight = lightPos - worldPos;
    float distanceToLight = length(toLight);
    
    if (distanceToLight < 0.001)
        return 1.0;
    
    // 构建从片元指向光源的射线
    RayWS shadowRay;
    shadowRay.Origin = worldPos;
    shadowRay.Direction = normalize(toLight);
    
#ifdef ENABLE_TRANSLUCENT_OBJECTS
    // === 半透明模式：收集多个交点，计算介质透射率 ===
    
    // BVH 射线求交（收集所有交点）
    IntersectsRaySegmentResultArray intersects;
    if (!IntersectRayBVHArray(shadowRay, distanceToLight, intersects))
    {
        // 无遮挡
        return 1.0;
    }
    
    // 使用 GetIntervals 构建介质区间
    RayMarchingInterval intervals[MAX_RAYMARCHING_INTERVALS];
    int intervalCount = 0;
    GetIntervals(shadowRay, intersects, distanceToLight, intervals, intervalCount);
    
    // 计算阴影衰减（基于介质距离和密度的指数衰减）
    float shadowAtten = 1.0;
    
    for (int i = 0; i < intervalCount; i++)
    {
        RayMarchingInterval interval = intervals[i];
        MaterialData mat = _BVH_Material_Buffer[interval.matIdx];
        
        // 计算介质区间距离
        float mediumDist = length(interval.end - interval.start);
        
        // 指数衰减：透射率 = exp(-distance * density)
        float segmentTransmittance = exp(-mediumDist * mat.Density);
        
        shadowAtten *= segmentTransmittance;
        
        // 完全遮挡时提前退出
        if (shadowAtten < 0.001)
            return 0.0;
    }
    
    return shadowAtten;
    
#else
    // === 不透明模式：只检查最近交点，有遮挡则完全阻挡 ===
    
    IntersectsRaySegmentResult hit;
    if (IntersectRayBVH(shadowRay, distanceToLight, hit))
    {
        // 有遮挡物，完全阻挡
        return 0.0;
    }
    
    // 无遮挡
    return 1.0;
    
#endif
}

/// <summary>
/// 计算从物体内部片元到光源的阴影衰减（忽略自阴影）
/// 专用于 sprite 内部像素的点光源计算
/// 启用半透明时：记录自身材质并排除，对其他材质计算介质透射率
/// 禁用半透明时：忽略第一个出点（自身边界），检查是否有其他入点遮挡
/// </summary>
/// <param name="worldPos">片元世界位置 (2D)，位于某个 sprite 内部</param>
/// <param name="lightPos">光源世界位置 (2D)</param>
/// <returns>阴影系数 [0, 1]，0 = 被其他物体完全遮挡，1 = 无遮挡</returns>
float CalculateShadowAttenuationInterior(float2 worldPos, float2 lightPos)
{
    float2 toLight = lightPos - worldPos;
    float distanceToLight = length(toLight);
    
    if (distanceToLight < 0.001)
        return 1.0;
    
    // 构建从片元指向光源的射线
    RayWS shadowRay;
    shadowRay.Origin = worldPos;
    shadowRay.Direction = normalize(toLight);
    
#ifdef ENABLE_TRANSLUCENT_OBJECTS
    // === 半透明模式：收集多个交点，计算介质透射率，排除自身材质 ===
    
    // BVH 射线求交
    IntersectsRaySegmentResultArray intersects;
    if (!IntersectRayBVHArray(shadowRay, distanceToLight, intersects))
    {
        // 无碰撞（理论上不应该发生，因为我们在物体内部）
        return 1.0;
    }
    
    // 找到第一个出点的材质索引（自身材质）
    // 出点 = dot(hitNormal, rayDir) > 0
    int selfMatIdx = -1;
    for (int k = 0; k < intersects.intersectsCount; k++)
    {
        IntersectsRaySegmentResult hit = intersects.results[k];
        float hitDist = distance(worldPos, hit.hitPoint);
        if (hitDist < 0.01 || hitDist > distanceToLight - 0.01)
            continue;
        
        float normalDotDir = dot(hit.hitNormal, shadowRay.Direction);
        if (normalDotDir > 0.0)
        {
            // 第一个出点就是自身边界
            selfMatIdx = hit.matIdx;
            break;
        }
    }
    
    // 构建介质区间并计算透射率
    // 使用入点-出点配对的方式，类似 RayMarchingInterval
    float shadowAtten = 1.0;
    
    // 追踪当前是否在某个介质内部，以及对应的入点信息
    // 使用简化的栈结构：假设最多嵌套 8 层
    #define MAX_MEDIUM_STACK 8
    float2 entryPoints[MAX_MEDIUM_STACK];
    int entryMatIdx[MAX_MEDIUM_STACK];
    int stackDepth = 0;
    
    for (int i = 0; i < intersects.intersectsCount; i++)
    {
        IntersectsRaySegmentResult hit = intersects.results[i];
        
        // 检查命中点是否在有效范围内
        float hitDist = distance(worldPos, hit.hitPoint);
        if (hitDist < 0.01 || hitDist > distanceToLight - 0.01)
            continue;
        
        // 如果是自身材质，跳过（自阴影排除）
        if (hit.matIdx == selfMatIdx)
            continue;
        
        // 判断是"出点"还是"入点"
        float normalDotDir = dot(hit.hitNormal, shadowRay.Direction);
        
        if (normalDotDir < 0.0)
        {
            // 入点：进入某个介质
            if (stackDepth < MAX_MEDIUM_STACK)
            {
                entryPoints[stackDepth] = hit.hitPoint;
                entryMatIdx[stackDepth] = hit.matIdx;
                stackDepth++;
            }
        }
        else
        {
            // 出点：离开某个介质
            // 查找对应的入点（相同材质）
            for (int j = stackDepth - 1; j >= 0; j--)
            {
                if (entryMatIdx[j] == hit.matIdx)
                {
                    // 找到配对的入点，计算介质距离和透射率
                    float2 entryPt = entryPoints[j];
                    float2 exitPt = hit.hitPoint;
                    float mediumDist = distance(entryPt, exitPt);
                    
                    MaterialData mat = _BVH_Material_Buffer[hit.matIdx];
                    float segmentTransmittance = exp(-mediumDist * mat.Density);
                    
                    shadowAtten *= segmentTransmittance;
                    
                    // 从栈中移除这个入点（通过将后面的元素前移）
                    for (int m = j; m < stackDepth - 1; m++)
                    {
                        entryPoints[m] = entryPoints[m + 1];
                        entryMatIdx[m] = entryMatIdx[m + 1];
                    }
                    stackDepth--;
                    break;
                }
            }
        }
        
        // 完全遮挡时提前退出
        if (shadowAtten < 0.001)
            return 0.0;
    }
    
    // 处理未配对的入点（介质延伸到光源）
    for (int n = 0; n < stackDepth; n++)
    {
        float2 entryPt = entryPoints[n];
        float mediumDist = distance(entryPt, lightPos);
        
        MaterialData mat = _BVH_Material_Buffer[entryMatIdx[n]];
        float segmentTransmittance = exp(-mediumDist * mat.Density);
        
        shadowAtten *= segmentTransmittance;
        
        if (shadowAtten < 0.001)
            return 0.0;
    }
    
    #undef MAX_MEDIUM_STACK
    
    return shadowAtten;
    
#else
    // === 不透明模式：忽略第一个出点（自身边界），检查是否有其他入点遮挡 ===
    
    // 使用单次求交函数
    IntersectsRaySegmentResult hit;
    if (!IntersectRayBVH(shadowRay, distanceToLight, hit))
    {
        // 无碰撞
        return 1.0;
    }
    
    // 检查是否为出点（自身边界）
    float normalDotDir = dot(hit.hitNormal, shadowRay.Direction);
    if (normalDotDir > 0.0)
    {
        // 第一个碰撞是出点（自身边界），需要继续检查后续是否有遮挡
        // 从出点位置继续发射射线
        RayWS continueRay;
        continueRay.Origin = hit.hitPoint + shadowRay.Direction * 0.02; // 略微偏移避免重复碰撞
        continueRay.Direction = shadowRay.Direction;
        
        float remainingDist = distanceToLight - distance(worldPos, hit.hitPoint) - 0.02;
        
        IntersectsRaySegmentResult nextHit;
        if (remainingDist > 0.01 && IntersectRayBVH(continueRay, remainingDist, nextHit))
        {
            // 后续有遮挡
            return 0.0;
        }
        
        // 无后续遮挡
        return 1.0;
    }
    else
    {
        // 第一个碰撞是入点（其他物体的遮挡）
        return 0.0;
    }
    
#endif
}

// 摄像机矩阵
float4x4 MatrixInvVP;
float4x4 MatrixVP;

// 像素坐标和世界空间的转换
float2 posPixel2World(float2 pixelPos, float2 screenParam)
{
    float2 uv = pixelPos / screenParam;
    float2 ndc = uv * 2.0 - 1.0;
    #if UNITY_UV_STARTS_AT_TOP
    ndc.y = -ndc.y;
    #endif
    float deviceDepth = 0.0;
    float4 clipPos = float4(ndc, deviceDepth, 1.0);
    float4 posWSRaw = mul(MatrixInvVP, clipPos);
    float2 posWS = posWSRaw.xy / posWSRaw.w;
    return posWS;
}
float2 posWorld2Pixel(float3 worldPos, float2 screenParam)
{
    // 1. 世界空间 -> 裁剪空间 (Clip Space)
    float4 clipPos = mul(MatrixVP, float4(worldPos, 1.0));
    // 也可以使用 URP 内置函数: float4 clipPos = TransformWorldToHClip(worldPos);

    // 2. 裁剪空间 -> NDC (-1 ~ 1)
    float2 ndc = clipPos.xy / clipPos.w;

    // 3. 处理平台差异 (Y轴翻转)
    // 逻辑与 forward 函数完全一致：
    // 如果之前为了匹配 NDC 翻转了 Y，现在为了变回屏幕 UV，需要再次翻转回来
    #if UNITY_UV_STARTS_AT_TOP
    ndc.y = -ndc.y;
    #endif

    // 4. NDC -> UV (0 ~ 1)
    float2 uv = ndc * 0.5 + 0.5;

    // 5. UV -> 屏幕像素坐标
    float2 pixelPos = uv * screenParam;

    return pixelPos;
}


// 辅助函数：判断一个点是否在Sprite内部
bool IsInsideSprite(float2 posWS)
{
    // 1. 构建一条任意方向的极短测试射线
    RayWS testRay;
    testRay.Origin = posWS;
    testRay.Direction = float2(1.0, 1.0);

    IntersectsRaySegmentResult result;
    if (IntersectRayBVH(testRay, result))
    {
        if (dot(result.hitNormal, testRay.Direction) > 0.0)
            return true;
    }
    return false;
}

TEXTURE2D(_RCWB_LightResult);
SAMPLER(sampler_RCWB_LightResult);
TEXTURE2D(_RCWB_LightResult_Blur);
SAMPLER(sampler_RCWB_LightResult_Blur);

TEXTURE2D(_RCWB_DirectionResult);
SAMPLER(sampler_RCWB_DirectionResult);
TEXTURE2D(_RCWB_DirectionResult_Blur);
SAMPLER(sampler_RCWB_DirectionResult_Blur);

// 向shader开放的函数
struct RcwbLightData{
    half3 color;
    half2 direction;
    bool hasDirection;
};

RcwbLightData GetRcwbLightData(float2 uv, float2 targetRenderSize, out bool isInsideSprite){
    // 先计算像素坐标和世界坐标
    float2 pixelPos = uv * targetRenderSize;
    float2 posWS = posPixel2World(pixelPos, targetRenderSize);
    isInsideSprite = IsInsideSprite(posWS);

    RcwbLightData data;

    if (!isInsideSprite)
    {
        data.color = SAMPLE_TEXTURE2D(_RCWB_LightResult, sampler_RCWB_LightResult, uv).rgb;
        data.direction = SAMPLE_TEXTURE2D(_RCWB_DirectionResult, sampler_RCWB_DirectionResult, uv);
    }
    else
    {
        data.color = SAMPLE_TEXTURE2D(_RCWB_LightResult_Blur, sampler_RCWB_LightResult_Blur, uv).rgb;
        data.direction = SAMPLE_TEXTURE2D(_RCWB_DirectionResult_Blur, sampler_RCWB_DirectionResult_Blur, uv);
    }
    // // 先全部采样纹理
    // half4 lightColor = SAMPLE_TEXTURE2D(_RCWB_LightResult, sampler_RCWB_LightResult, uv);
    // half2 direction = SAMPLE_TEXTURE2D(_RCWB_DirectionResult, sampler_RCWB_DirectionResult, uv);

    // data.color = lightColor.rgb;
    // data.direction = direction;
    data.hasDirection = length(data.direction) > 0.0001;

    return data;
}

#endif