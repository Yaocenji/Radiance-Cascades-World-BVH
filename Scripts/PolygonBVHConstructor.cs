using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;

namespace RadianceCascadesWorldBVH
{
    public class PolygonBVHConstructor
    {
        // 共享引用：需要传递给GPU的数组
        private List<edgeBVH> edges;
        private List<SpriteRenderer> spriteRenderers;
        
        // BVH结果数据（对外暴露，PolygonManager需要用于GPU上传和绘制）
        public LBVHNodeRaw[] nodes;
        public int rootNodeIndex;
        
        // BVH数据加入GPU
        public List<LBVHNodeGpu> gpuNodes;
        
        // BVH中间数据（仅在构建过程中使用）
        private List<uint> mortonCodes = new List<uint>();
        private List<int> index = new List<int>();
        
        // 重拍node
        private LBVHNodeRaw[] _tempSortedNodes; // 用于暂存重排后的节点
        private int[] _indexMap;             // 用于映射 OldIndex -> NewIndex
        private int[] _bfsQueue;             // 用数组模拟队列
        
        public PolygonBVHConstructor(List<edgeBVH> edges, List<SpriteRenderer> spriteRenderers)
        {
            this.edges = edges;
            this.spriteRenderers = spriteRenderers;
        }
        
        /// <summary>
        /// 辅助函数：位展开
        /// </summary>
        private static uint ExpandBits(uint v)
        {
            v &= 0x0000FFFF;                      // 确保只取低 16 位
            v = (v | (v << 8)) & 0x00FF00FFu;     // 111111110000000011111111
            v = (v | (v << 4)) & 0x0F0F0F0Fu;     // 11110000111100001111000011110000
            v = (v | (v << 2)) & 0x33333333u;     // 11001100110011001100110011001100
            v = (v | (v << 1)) & 0x55555555u;     // 10101010101010101010101010101010
            return v;
        }
        
        /// <summary>
        /// 辅助函数：morton code计算
        /// </summary>
        public static uint MortonCode(Vector2 position, Vector2 aabbMin, Vector2 aabbMax)
        {
            Vector2 normalizedPos = (position - aabbMin) / (aabbMax - aabbMin);
            // 1. 安全钳制到 0-1，防止越界导致溢出
            float x = Mathf.Clamp01(normalizedPos.x);
            float y = Mathf.Clamp01(normalizedPos.y);

            // 2. 量化：映射到 [0, 65535]
            uint ix = (uint)(x * 65535.0f);
            uint iy = (uint)(y * 65535.0f);

            // 3. 交叉合并位
            // x 放偶数位，y 放奇数位 (或者反过来)
            uint xx = ExpandBits(ix);
            uint yy = ExpandBits(iy);

            return xx | (yy << 1);
        }
        
        
        // 计算前导零个数的辅助函数
    // .NET Core 3.0+ 可以直接使用 System.Numerics.BitOperations.LeadingZeroCount(x)
    // 为了兼容性，这里提供一个通用实现
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int CountLeadingZeros(uint x)
        {
            if (x == 0) return 32;
#if NETCOREAPP3_0_OR_GREATER
    return System.Numerics.BitOperations.LeadingZeroCount(x);
#else
            // 简单的去重算法或查表法，这里使用简单的二分法
            int n = 0;
            if (x <= 0x0000FFFF) { n = n + 16; x = x << 16; }
            if (x <= 0x00FFFFFF) { n = n + 8; x = x << 8; }
            if (x <= 0x0FFFFFFF) { n = n + 4; x = x << 4; }
            if (x <= 0x3FFFFFFF) { n = n + 2; x = x << 2; }
            if (x <= 0x7FFFFFFF) { n = n + 1; }
            return n;
#endif
        }

        // 计算索引 i 和 j 处莫顿码的最长公共前缀长度
        private int GetLongestCommonPrefix(int i, int j, int numPrimitives)
        {
            // 1. 边界检查
            if (j < 0 || j >= numPrimitives) return -1;

            uint codeI = mortonCodes[i];
            uint codeJ = mortonCodes[j];

            if (codeI != codeJ)
            {
                // 如果码不同，计算 XOR 后的前导零
                return CountLeadingZeros(codeI ^ codeJ);
            }
            else
            {
                // 如果码相同，使用索引作为 Tie-breaker (打破平局)
                // 加上 32 (uint位数) 保证比任何不相同的情况都大
                return 32 + CountLeadingZeros((uint)i ^ (uint)j);
            }
        }
        

        /// <summary>
        ///  获取边
        /// </summary>
        public void GetBvhEdges()
        {
            // 先清空
            edges.Clear();
            
            // 直接暴力运行时生成，先跑通算法
            for (int sidx = 0; sidx < spriteRenderers.Count; sidx++)
            {
                SpriteRenderer spriteRenderer = spriteRenderers[sidx];
                // 先获取环的数量
                int loopCount = spriteRenderer.sprite.GetPhysicsShapeCount();
                for (int i = 0; i < loopCount; i++)
                {
                    // 环数据
                    List<Vector2> points = new List<Vector2>();
                    // 环的点数目
                    int pointCount = spriteRenderer.sprite.GetPhysicsShape(i, points);
                    
                    // 暂时不管顺反问题
                    // TODO
                    
                    // 应用transform
                    for (int j = 0; j < pointCount; j++)
                    {
                        points[j] = spriteRenderer.transform.TransformPoint(points[j].x, points[j].y, 0f);
                    }
                    
                    // 将环转化为边集合
                    for (int j = 0; j < pointCount; j++)
                    {
                        edgeBVH edge = new edgeBVH();
                        edge.start = points[j % pointCount];
                        edge.end = points[(j + 1) % pointCount];
                        edge.matIdx = sidx;
                        edges.Add(edge);
                    }
                }
            }
        }

        /// <summary>
        /// 计算所有边的莫顿码
        /// </summary>
        public void CalculateMortonCodes(Vector4 sceneAABB)
        {
            // 重新生成莫顿码和索引
            mortonCodes.Clear();
            index.Clear();
            
            for (int i = 0; i < edges.Count; i++)
            {
                edgeBVH edge = edges[i];
                // 计算中点
                Vector2 center = (edge.end + edge.start) / 2.0f;
                // 计算莫顿码
                uint mortonCode = MortonCode(center, new Vector2(sceneAABB.x, sceneAABB.y), new Vector2(sceneAABB.z, sceneAABB.w));
                // 添加到莫顿码集合
                mortonCodes.Add(mortonCode);
                // 添加到索引集合
                index.Add(i);
            }
        }
        
        /// <summary>
        /// 排序莫顿码
        /// </summary>
        public void SortMortonCodes()
        {
            // 1. 将 List 转换为 Array
            // Array.Sort 针对数组进行了极致优化，比 List.Sort 更底层
            uint[] codesArray = mortonCodes.ToArray();
            int[] indexArray = index.ToArray();

            // 2. 执行双数组排序 (核心步骤)
            // 参数1 (Keys): 依据这个数组的值进行排序 (你的 Morton Code)
            // 参数2 (Items): 这个数组的元素会跟随 Keys 自动换位 (你的 Index)
            Array.Sort(codesArray, indexArray);

            // 3. (可选) 如果你后续逻辑必须用 List，可以转回去
            // 但通常建议直接用 Array 继续后面的 BVH 构建，因为性能更好
            mortonCodes = new List<uint>(codesArray);
            index = new List<int>(indexArray);
    
        }

        
        /// <summary>
        /// 生成层次结构
        /// </summary>
        public void BuildBVHStructure()
        {
            int numPrimitives = mortonCodes.Count;
            if (numPrimitives == 0) return;

            // 分配节点内存：N个叶子 + N-1个内部节点
            int numNodes = 2 * numPrimitives - 1;
            nodes = new LBVHNodeRaw[numNodes];

            // 初始化所有节点的父节点为 -1
            for (int i = 0; i < numNodes; i++)
            {
                nodes[i].Parent = -1;
                nodes[i].LeftChild = -1;
                nodes[i].RightChild = -1;
                nodes[i].ObjectIndex = -1;
            }

            // 1. 并行循环的串行化：处理每一个内部节点
            // 内部节点由索引 i (0 到 numPrimitives - 2) 生成
            // 在 nodes 数组中，它们位于 numPrimitives + i
            for (int i = 0; i < numPrimitives - 1; i++)
            {
                // --- 下面是 Karras 算法的核心逻辑 ---

                // 确定方向 d (+1 或 -1)
                int d = (GetLongestCommonPrefix(i, i + 1, numPrimitives) - GetLongestCommonPrefix(i, i - 1, numPrimitives)) > 0 ? 1 : -1;

                // 计算当前节点 i 的最小 LCP (Longest Common Prefix)
                int minDelta = GetLongestCommonPrefix(i, i - d, numPrimitives);

                // 确定范围的另一端 l_max
                int lMax = 2;
                while (GetLongestCommonPrefix(i, i + lMax * d, numPrimitives) > minDelta)
                {
                    lMax *= 2;
                }

                // 二分查找精确的另一端 j
                int l = 0;
                for (int t = lMax / 2; t >= 1; t /= 2)
                {
                    if (GetLongestCommonPrefix(i, i + (l + t) * d, numPrimitives) > minDelta)
                    {
                        l += t;
                    }
                }
                int j = i + l * d;

                // 寻找分割点 gamma
                int deltaNode = GetLongestCommonPrefix(i, j, numPrimitives);
                int s = 0;
                int first = Math.Min(i, j);
                int last = Math.Max(i, j);
                
                int split = first;
                int step = last - first;

                do
                {
                    step = (step + 1) >> 1; // ceil(step / 2)
                    int newSplit = split + step;
                    if (newSplit < last)
                    {
                        if (GetLongestCommonPrefix(first, newSplit, numPrimitives) > deltaNode)
                        {
                            split = newSplit;
                        }
                    }
                } while (step > 1);

                // split 就是 gamma，范围被分为 [first, split] 和 [split+1, last]

                // --- 构建拓扑连接 ---

                // 当前内部节点在数组中的真实索引
                int currentNodeIdx = numPrimitives + i;

                int leftIdx = split;
                int rightIdx = split + 1;

                // 处理左孩子
                // 如果左范围只包含一个元素，则是叶子；否则是内部节点
                // 在我们的数组映射中：
                //   叶子索引直接是 [0...N-1]
                //   内部节点索引是 N + 逻辑索引
                // Karras算法有个特性：内部节点 k 负责分割以 k 开始的范围。
                // 所以如果孩子是内部节点，它的逻辑索引就是 leftIdx (即 split)
                
                int leftChildNodeIdx;
                if (Math.Min(i, j) == leftIdx)
                {
                    // 左孩子是叶子
                    leftChildNodeIdx = leftIdx; 
                }
                else
                {
                    // 左孩子是内部节点
                    leftChildNodeIdx = numPrimitives + leftIdx; 
                }

                // 处理右孩子
                int rightChildNodeIdx;
                if (Math.Max(i, j) == rightIdx)
                {
                    // 右孩子是叶子
                    rightChildNodeIdx = rightIdx;
                }
                else
                {
                    // 右孩子是内部节点
                    rightChildNodeIdx = numPrimitives + rightIdx;
                }

                // 建立连接
                nodes[currentNodeIdx].LeftChild = leftChildNodeIdx;
                nodes[currentNodeIdx].RightChild = rightChildNodeIdx;

                nodes[leftChildNodeIdx].Parent = currentNodeIdx;
                nodes[rightChildNodeIdx].Parent = currentNodeIdx;
            }

            // 2. 寻找根节点
            // 根节点是唯一一个 Parent 仍为 -1 的内部节点
            rootNodeIndex = -1;
            for (int i = numPrimitives; i < numNodes; i++)
            {
                if (nodes[i].Parent == -1)
                {
                    rootNodeIndex = i;
                    break;
                }
            }
        }
        
        
        /// <summary>
        /// 计算所有节点的包围盒
        /// </summary>
        // 重新计算所有节点的AABB
        public void RefitBVH()
        {
            if (rootNodeIndex == -1) return;
    
            // 从根节点开始递归计算
            CalculateAABBRecursive(rootNodeIndex);
        }

        
        // 递归函数
        private void CalculateAABBRecursive(int nodeIdx)
        {
            // 引用类型技巧：C#数组直接取结构体是值拷贝，修改不会生效。
            // 所以我们需要直接操作数组引用，或者在修改后赋值回去。
            // 这里使用 ref 变量（如果C#版本支持 ref locals）或者直接操作 nodes[nodeIdx]

            // 1. 如果是叶子节点
            // 注意：在我们的布局中，索引 < mortonCodes.Count 的都是叶子
            int numPrimitives = mortonCodes.Count;
            if (nodeIdx < numPrimitives)
            {
                // 获取排序后的对象ID
                int originalObjIdx = index[nodeIdx];
        
                // 记录原始索引以便查询时使用
                nodes[nodeIdx].ObjectIndex = originalObjIdx; 
        
                // 获取原始几何体（边）
                edgeBVH edge = edges[originalObjIdx];

                // 计算边的 AABB
                nodes[nodeIdx].Min = Vector2.Min(edge.start, edge.end) - Vector2.one * .01f;
                nodes[nodeIdx].Max = Vector2.Max(edge.start, edge.end) + Vector2.one * .01f;
        
                // 确保叶子没有孩子
                nodes[nodeIdx].LeftChild = -1;
                nodes[nodeIdx].RightChild = -1;
        
                return;
            }

            // 2. 如果是内部节点
            int leftChild = nodes[nodeIdx].LeftChild;
            int rightChild = nodes[nodeIdx].RightChild;

            // 递归计算子节点
            if (leftChild != -1) CalculateAABBRecursive(leftChild);
            if (rightChild != -1) CalculateAABBRecursive(rightChild);

            // 合并子节点的 AABB
            Vector2 minL = nodes[leftChild].Min;
            Vector2 maxL = nodes[leftChild].Max;
            Vector2 minR = nodes[rightChild].Min;
            Vector2 maxR = nodes[rightChild].Max;

            nodes[nodeIdx].Min = Vector2.Min(minL, minR);
            nodes[nodeIdx].Max = Vector2.Max(maxL, maxR);
    
            // 内部节点不指向具体对象
            nodes[nodeIdx].ObjectIndex = -1; 
        }
        
        
        /// <summary>
        /// 执行 BFS 重排，将树在内存中线性化
        /// </summary>
        public void ReorderBVHToBFS()
        {
            if (rootNodeIndex == -1 || nodes == null || nodes.Length == 0) return;

            int nodeCount = nodes.Length;

            // 1. 确保缓存数组容量足够 (按 1.2 倍扩容防止频繁分配)
            if (_tempSortedNodes == null || _tempSortedNodes.Length < nodeCount)
            {
                int newSize = Mathf.NextPowerOfTwo(nodeCount); // 或者 * 1.5
                _tempSortedNodes = new LBVHNodeRaw[newSize];
                _indexMap = new int[newSize];
                _bfsQueue = new int[newSize];
            }

            // 2. 初始化 BFS 状态
            int queueHead = 0; // 队头
            int queueTail = 0; // 队尾
            int newIndexCounter = 0;

            // 将根节点入队
            _bfsQueue[queueTail++] = rootNodeIndex;

            // 3. 第一轮 BFS：生成映射表 (Old -> New) 并确定新顺序
            // 这里我们只记录顺序，还没拷贝数据，因为修正指针需要先知道所有孩子的新位置
            // 但其实我们可以一边拷贝一边修？不行，孩子还没生成。
            // 最高效的方法：双指针遍历。
            
            // 既然是 BFS，新数组的第 i 个元素，就是 BFS 队列弹出的第 i 个元素。
            // 所以 _bfsQueue 其实就是 sortedNodes 的来源索引数组。
            
            while (queueHead < queueTail)
            {
                int oldIdx = _bfsQueue[queueHead++]; // Dequeue
                
                // 记录映射：旧索引 oldIdx 在新数组中变成了 newIndexCounter
                _indexMap[oldIdx] = newIndexCounter;
                newIndexCounter++;

                // 将孩子入队
                int left = nodes[oldIdx].LeftChild;
                int right = nodes[oldIdx].RightChild;

                if (left != -1) _bfsQueue[queueTail++] = left;
                if (right != -1) _bfsQueue[queueTail++] = right;
            }

            // 4. 第二轮：填充数据并修正指针
            // 直接遍历刚才生成的队列顺序（它保存了所有节点的旧索引，且顺序就是 BFS 顺序）
            for (int i = 0; i < nodeCount; i++)
            {
                int oldIdx = _bfsQueue[i];
                
                // 结构体值拷贝
                LBVHNodeRaw nodeRaw = nodes[oldIdx];

                // 修正左右孩子索引
                if (nodeRaw.LeftChild != -1)
                    nodeRaw.LeftChild = _indexMap[nodeRaw.LeftChild]; // O(1) 查表
                
                if (nodeRaw.RightChild != -1)
                    nodeRaw.RightChild = _indexMap[nodeRaw.RightChild]; // O(1) 查表
                    
                // Parent 和 ObjectIndex 不需要变 (Parent 其实变了，但渲染通常不需要 Parent)
                // 如果你需要 Parent，也可以修：if (nodeRaw.Parent != -1) nodeRaw.Parent = _indexMap[nodeRaw.Parent];

                _tempSortedNodes[i] = nodeRaw;
            }

            // 5. 交换引用 (这一步几乎无消耗)
            // 注意：这里我们把缓存数组和主数组交换了，
            // 下一帧 nodes 就会变成缓存，_tempSortedNodes 变成旧的 nodes 被复用。
            // 这样避免了 Array.Copy 的开销。
            var swap = nodes;
            nodes = _tempSortedNodes; // nodes 现在指向了 BFS 排序好的数据
            _tempSortedNodes = swap;  // 缓存区拿走旧数组以备下帧使用

            // 6. 更新根节点索引
            rootNodeIndex = 0; // BFS 保证根节点永远在 0
        }
    }
}
