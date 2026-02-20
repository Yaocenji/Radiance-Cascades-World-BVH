using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace RadianceCascadesWorldBVH
{
    /// <summary>
    /// 使用 Burst + Unity Job System 加速的 BVH 构建器
    /// 逻辑与 PolygonBVHConstructor 完全一致，仅做并行化优化
    /// </summary>
    public class PolygonBVHConstructorAccelerated : IDisposable
    {
        // 共享引用：需要传递给GPU的数组
        private List<edgeBVH> edges;
        private List<SpriteRenderer> spriteRenderers;
        
        // BVH结果数据（对外暴露，PolygonManager需要用于GPU上传和绘制）
        public LBVHNodeRaw[] nodes;
        public int rootNodeIndex;
        
        // GPU 打包后的 BVH 数据
        public LBVHNodeGpu[] gpuNodes;
        
        // Native 容器（用于 Job System）
        private NativeArray<edgeBVH> _nativeEdges;
        private NativeArray<uint> _nativeMortonCodes;
        private NativeArray<int> _nativeIndices;
        private NativeArray<LBVHNodeRaw> _nativeNodes;
        private NativeArray<LBVHNodeGpu> _nativeGpuNodes;
        
        // BFS 重排用的临时数组
        private NativeArray<LBVHNodeRaw> _tempSortedNodes;
        private NativeArray<int> _indexMap;
        private NativeArray<int> _bfsQueue;
        
        // 当前分配的容量
        private int _allocatedEdgeCapacity;
        private int _allocatedNodeCapacity;
        
        // 是否已释放
        private bool _disposed;
        
        public PolygonBVHConstructorAccelerated(List<edgeBVH> edges, List<SpriteRenderer> spriteRenderers)
        {
            this.edges = edges;
            this.spriteRenderers = spriteRenderers;
            _allocatedEdgeCapacity = 0;
            _allocatedNodeCapacity = 0;
            _disposed = false;
        }
        
        /// <summary>
        /// 确保 Native 容器容量足够
        /// </summary>
        private void EnsureEdgeCapacity(int requiredCapacity)
        {
            if (_allocatedEdgeCapacity >= requiredCapacity) return;
            
            int newCapacity = Mathf.NextPowerOfTwo(requiredCapacity);
            
            // 释放旧的
            if (_nativeEdges.IsCreated) _nativeEdges.Dispose();
            if (_nativeMortonCodes.IsCreated) _nativeMortonCodes.Dispose();
            if (_nativeIndices.IsCreated) _nativeIndices.Dispose();
            
            // 分配新的
            _nativeEdges = new NativeArray<edgeBVH>(newCapacity, Allocator.Persistent);
            _nativeMortonCodes = new NativeArray<uint>(newCapacity, Allocator.Persistent);
            _nativeIndices = new NativeArray<int>(newCapacity, Allocator.Persistent);
            
            _allocatedEdgeCapacity = newCapacity;
        }
        
        private void EnsureNodeCapacity(int requiredCapacity)
        {
            if (_allocatedNodeCapacity >= requiredCapacity) return;
            
            int newCapacity = Mathf.NextPowerOfTwo(requiredCapacity);
            
            // 释放旧的
            if (_nativeNodes.IsCreated) _nativeNodes.Dispose();
            if (_nativeGpuNodes.IsCreated) _nativeGpuNodes.Dispose();
            if (_tempSortedNodes.IsCreated) _tempSortedNodes.Dispose();
            if (_indexMap.IsCreated) _indexMap.Dispose();
            if (_bfsQueue.IsCreated) _bfsQueue.Dispose();
            
            // 分配新的
            _nativeNodes = new NativeArray<LBVHNodeRaw>(newCapacity, Allocator.Persistent);
            _nativeGpuNodes = new NativeArray<LBVHNodeGpu>(newCapacity, Allocator.Persistent);
            _tempSortedNodes = new NativeArray<LBVHNodeRaw>(newCapacity, Allocator.Persistent);
            _indexMap = new NativeArray<int>(newCapacity, Allocator.Persistent);
            _bfsQueue = new NativeArray<int>(newCapacity, Allocator.Persistent);
            
            _allocatedNodeCapacity = newCapacity;
        }
        
        /// <summary>
        /// 获取边（此方法涉及 Unity API，无法完全并行化，但数据转换可以优化）
        /// </summary>
        public void GetBvhEdges()
        {
            edges.Clear();
            
            for (int sidx = 0; sidx < spriteRenderers.Count; sidx++)
            {
                SpriteRenderer spriteRenderer = spriteRenderers[sidx];
                int loopCount = spriteRenderer.sprite.GetPhysicsShapeCount();
                
                for (int i = 0; i < loopCount; i++)
                {
                    List<Vector2> points = new List<Vector2>();
                    int pointCount = spriteRenderer.sprite.GetPhysicsShape(i, points);
                    
                    for (int j = 0; j < pointCount; j++)
                    {
                        points[j] = spriteRenderer.transform.TransformPoint(points[j].x, points[j].y, 0f);
                    }
                    
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
        /// 计算所有边的莫顿码（并行化）
        /// </summary>
        public void CalculateMortonCodes(Vector4 sceneAABB)
        {
            int edgeCount = edges.Count;
            if (edgeCount == 0) return;
            
            EnsureEdgeCapacity(edgeCount);
            
            // 拷贝边数据到 NativeArray
            for (int i = 0; i < edgeCount; i++)
            {
                _nativeEdges[i] = edges[i];
            }
            
            // 创建并调度 Job
            var job = new CalculateMortonCodesJob
            {
                Edges = _nativeEdges,
                MortonCodes = _nativeMortonCodes,
                Indices = _nativeIndices,
                AABBMin = new float2(sceneAABB.x, sceneAABB.y),
                AABBMax = new float2(sceneAABB.z, sceneAABB.w),
                EdgeCount = edgeCount
            };
            
            JobHandle handle = job.Schedule(edgeCount, 64);
            handle.Complete();
        }
        
        /// <summary>
        /// 排序莫顿码（使用 NativeSortExtension）
        /// </summary>
        public void SortMortonCodes()
        {
            int edgeCount = edges.Count;
            if (edgeCount == 0) return;
            
            // 创建临时数组用于排序
            var sortKeys = new NativeArray<ulong>(edgeCount, Allocator.TempJob);
            
            // 将 Morton Code 和 Index 打包成 ulong 进行排序（高32位是 Morton，低32位是 Index）
            var packJob = new PackSortKeysJob
            {
                MortonCodes = _nativeMortonCodes,
                Indices = _nativeIndices,
                SortKeys = sortKeys,
                Count = edgeCount
            };
            packJob.Schedule(edgeCount, 64).Complete();
            
            // 并行排序
            sortKeys.Sort();
            
            // 解包排序结果
            var unpackJob = new UnpackSortKeysJob
            {
                SortKeys = sortKeys,
                MortonCodes = _nativeMortonCodes,
                Indices = _nativeIndices,
                Count = edgeCount
            };
            unpackJob.Schedule(edgeCount, 64).Complete();
            
            sortKeys.Dispose();
        }
        
        /// <summary>
        /// 生成层次结构（Karras 算法，可并行化）
        /// </summary>
        public void BuildBVHStructure()
        {
            int numPrimitives = edges.Count;
            if (numPrimitives == 0) return;
            
            int numNodes = 2 * numPrimitives - 1;
            EnsureNodeCapacity(numNodes);
            
            // 初始化节点（并行）
            var initJob = new InitNodesJob
            {
                Nodes = _nativeNodes,
                NodeCount = numNodes
            };
            initJob.Schedule(numNodes, 64).Complete();
            
            // 构建 BVH 拓扑（Karras 算法，每个内部节点独立处理，可并行）
            if (numPrimitives > 1)
            {
                var buildJob = new BuildBVHTopologyJob
                {
                    MortonCodes = _nativeMortonCodes,
                    Nodes = _nativeNodes,
                    NumPrimitives = numPrimitives
                };
                buildJob.Schedule(numPrimitives - 1, 64).Complete();
            }
            
            // 寻找根节点（串行，因为只需要找一个）
            rootNodeIndex = numPrimitives; // Karras 算法中，根节点通常是 numPrimitives
            
            // 验证根节点
            for (int i = numPrimitives; i < numNodes; i++)
            {
                if (_nativeNodes[i].Parent == -1)
                {
                    rootNodeIndex = i;
                    break;
                }
            }
        }
        
        /// <summary>
        /// 计算所有节点的包围盒（自底向上，使用迭代）
        /// </summary>
        public void RefitBVH()
        {
            if (rootNodeIndex == -1) return;
            
            int numPrimitives = edges.Count;
            int numNodes = _nativeNodes.Length;
            
            // 第一步：并行计算所有叶子节点的 AABB
            var leafJob = new ComputeLeafAABBJob
            {
                Nodes = _nativeNodes,
                Edges = _nativeEdges,
                Indices = _nativeIndices,
                NumPrimitives = numPrimitives
            };
            leafJob.Schedule(numPrimitives, 64).Complete();
            
            // 第二步：自底向上计算内部节点的 AABB
            // 使用原子计数器实现无锁自底向上遍历
            var atomicCounters = new NativeArray<int>(numNodes, Allocator.TempJob);
            
            var internalJob = new ComputeInternalAABBJob
            {
                Nodes = _nativeNodes,
                AtomicCounters = atomicCounters,
                NumPrimitives = numPrimitives
            };
            // 从所有叶子节点开始向上传播
            internalJob.Schedule(numPrimitives, 64).Complete();
            
            atomicCounters.Dispose();
        }
        
        /// <summary>
        /// 执行 BFS 重排（此步骤串行，但数据拷贝并行）
        /// </summary>
        public void ReorderBVHToBFS()
        {
            if (rootNodeIndex == -1) return;
            
            int nodeCount = 2 * edges.Count - 1;
            if (nodeCount <= 0) return;
            
            // BFS 遍历生成新顺序（串行）
            int queueHead = 0;
            int queueTail = 0;
            
            _bfsQueue[queueTail++] = rootNodeIndex;
            
            int newIndexCounter = 0;
            while (queueHead < queueTail)
            {
                int oldIdx = _bfsQueue[queueHead++];
                _indexMap[oldIdx] = newIndexCounter++;
                
                var node = _nativeNodes[oldIdx];
                if (node.LeftChild != -1) _bfsQueue[queueTail++] = node.LeftChild;
                if (node.RightChild != -1) _bfsQueue[queueTail++] = node.RightChild;
            }
            
            // 并行拷贝并修正指针
            var reorderJob = new ReorderNodesJob
            {
                SourceNodes = _nativeNodes,
                DestNodes = _tempSortedNodes,
                BfsQueue = _bfsQueue,
                IndexMap = _indexMap,
                NodeCount = nodeCount
            };
            reorderJob.Schedule(nodeCount, 64).Complete();
            
            // 交换数组
            var swap = _nativeNodes;
            _nativeNodes = _tempSortedNodes;
            _tempSortedNodes = swap;
            
            rootNodeIndex = 0;
        }
        
        /// <summary>
        /// 打包 GPU 节点（并行）
        /// </summary>
        public void PackGpuNodes()
        {
            int nodeCount = 2 * edges.Count - 1;
            if (nodeCount <= 0)
            {
                gpuNodes = null;
                return;
            }
            
            var packJob = new PackGpuNodesJob
            {
                Nodes = _nativeNodes,
                Edges = _nativeEdges,
                GpuNodes = _nativeGpuNodes,
                NodeCount = nodeCount
            };
            packJob.Schedule(nodeCount, 64).Complete();
            
            // 拷贝结果到托管数组
            if (gpuNodes == null || gpuNodes.Length < nodeCount)
            {
                gpuNodes = new LBVHNodeGpu[Mathf.NextPowerOfTwo(nodeCount)];
            }
            NativeArray<LBVHNodeGpu>.Copy(_nativeGpuNodes, 0, gpuNodes, 0, nodeCount);
            
            // 同时拷贝 nodes 到托管数组
            if (nodes == null || nodes.Length < nodeCount)
            {
                nodes = new LBVHNodeRaw[Mathf.NextPowerOfTwo(nodeCount)];
            }
            NativeArray<LBVHNodeRaw>.Copy(_nativeNodes, 0, nodes, 0, nodeCount);
        }
        
        public int GpuNodeCount => edges?.Count > 0 ? 2 * edges.Count - 1 : 0;
        
        public void Dispose()
        {
            if (_disposed) return;
            
            if (_nativeEdges.IsCreated) _nativeEdges.Dispose();
            if (_nativeMortonCodes.IsCreated) _nativeMortonCodes.Dispose();
            if (_nativeIndices.IsCreated) _nativeIndices.Dispose();
            if (_nativeNodes.IsCreated) _nativeNodes.Dispose();
            if (_nativeGpuNodes.IsCreated) _nativeGpuNodes.Dispose();
            if (_tempSortedNodes.IsCreated) _tempSortedNodes.Dispose();
            if (_indexMap.IsCreated) _indexMap.Dispose();
            if (_bfsQueue.IsCreated) _bfsQueue.Dispose();
            
            _disposed = true;
        }
        
        ~PolygonBVHConstructorAccelerated()
        {
            Dispose();
        }
    }
    
    // =========================================================
    // Burst Jobs
    // =========================================================
    
    [BurstCompile]
    struct CalculateMortonCodesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<edgeBVH> Edges;
        [WriteOnly] public NativeArray<uint> MortonCodes;
        [WriteOnly] public NativeArray<int> Indices;
        public float2 AABBMin;
        public float2 AABBMax;
        public int EdgeCount;
        
        public void Execute(int i)
        {
            if (i >= EdgeCount) return;
            
            edgeBVH edge = Edges[i];
            float2 center = (new float2(edge.start.x, edge.start.y) + new float2(edge.end.x, edge.end.y)) * 0.5f;
            
            MortonCodes[i] = ComputeMortonCode(center, AABBMin, AABBMax);
            Indices[i] = i;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static uint ExpandBits(uint v)
        {
            v &= 0x0000FFFF;
            v = (v | (v << 8)) & 0x00FF00FFu;
            v = (v | (v << 4)) & 0x0F0F0F0Fu;
            v = (v | (v << 2)) & 0x33333333u;
            v = (v | (v << 1)) & 0x55555555u;
            return v;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static uint ComputeMortonCode(float2 position, float2 aabbMin, float2 aabbMax)
        {
            float2 normalizedPos = (position - aabbMin) / (aabbMax - aabbMin);
            float x = math.clamp(normalizedPos.x, 0f, 1f);
            float y = math.clamp(normalizedPos.y, 0f, 1f);
            
            uint ix = (uint)(x * 65535.0f);
            uint iy = (uint)(y * 65535.0f);
            
            uint xx = ExpandBits(ix);
            uint yy = ExpandBits(iy);
            
            return xx | (yy << 1);
        }
    }
    
    [BurstCompile]
    struct PackSortKeysJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> MortonCodes;
        [ReadOnly] public NativeArray<int> Indices;
        [WriteOnly] public NativeArray<ulong> SortKeys;
        public int Count;
        
        public void Execute(int i)
        {
            if (i >= Count) return;
            SortKeys[i] = ((ulong)MortonCodes[i] << 32) | (uint)Indices[i];
        }
    }
    
    [BurstCompile]
    struct UnpackSortKeysJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<ulong> SortKeys;
        [WriteOnly] public NativeArray<uint> MortonCodes;
        [WriteOnly] public NativeArray<int> Indices;
        public int Count;
        
        public void Execute(int i)
        {
            if (i >= Count) return;
            ulong key = SortKeys[i];
            MortonCodes[i] = (uint)(key >> 32);
            Indices[i] = (int)(key & 0xFFFFFFFF);
        }
    }
    
    [BurstCompile]
    struct InitNodesJob : IJobParallelFor
    {
        public NativeArray<LBVHNodeRaw> Nodes;
        public int NodeCount;
        
        public void Execute(int i)
        {
            if (i >= NodeCount) return;
            
            LBVHNodeRaw node = new LBVHNodeRaw();
            node.Parent = -1;
            node.LeftChild = -1;
            node.RightChild = -1;
            node.ObjectIndex = -1;
            Nodes[i] = node;
        }
    }
    
    [BurstCompile]
    struct BuildBVHTopologyJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> MortonCodes;
        [NativeDisableParallelForRestriction]
        public NativeArray<LBVHNodeRaw> Nodes;
        public int NumPrimitives;
        
        public void Execute(int i)
        {
            // Karras 2012 算法
            int d = (GetLCP(i, i + 1) - GetLCP(i, i - 1)) > 0 ? 1 : -1;
            int minDelta = GetLCP(i, i - d);
            
            int lMax = 2;
            while (GetLCP(i, i + lMax * d) > minDelta)
                lMax *= 2;
            
            int l = 0;
            for (int t = lMax / 2; t >= 1; t /= 2)
            {
                if (GetLCP(i, i + (l + t) * d) > minDelta)
                    l += t;
            }
            int j = i + l * d;
            
            int deltaNode = GetLCP(i, j);
            int first = math.min(i, j);
            int last = math.max(i, j);
            
            int split = first;
            int step = last - first;
            do
            {
                step = (step + 1) >> 1;
                int newSplit = split + step;
                if (newSplit < last && GetLCP(first, newSplit) > deltaNode)
                    split = newSplit;
            } while (step > 1);
            
            int currentNodeIdx = NumPrimitives + i;
            
            int leftChildIdx = (math.min(i, j) == split) ? split : (NumPrimitives + split);
            int rightChildIdx = (math.max(i, j) == split + 1) ? (split + 1) : (NumPrimitives + split + 1);
            
            // 写入当前节点
            var node = Nodes[currentNodeIdx];
            node.LeftChild = leftChildIdx;
            node.RightChild = rightChildIdx;
            Nodes[currentNodeIdx] = node;
            
            // 写入子节点的 Parent
            var leftNode = Nodes[leftChildIdx];
            leftNode.Parent = currentNodeIdx;
            Nodes[leftChildIdx] = leftNode;
            
            var rightNode = Nodes[rightChildIdx];
            rightNode.Parent = currentNodeIdx;
            Nodes[rightChildIdx] = rightNode;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int GetLCP(int i, int j)
        {
            if (j < 0 || j >= NumPrimitives) return -1;
            
            uint codeI = MortonCodes[i];
            uint codeJ = MortonCodes[j];
            
            if (codeI != codeJ)
                return math.lzcnt(codeI ^ codeJ);
            else
                return 32 + math.lzcnt((uint)i ^ (uint)j);
        }
    }
    
    [BurstCompile]
    struct ComputeLeafAABBJob : IJobParallelFor
    {
        public NativeArray<LBVHNodeRaw> Nodes;
        [ReadOnly] public NativeArray<edgeBVH> Edges;
        [ReadOnly] public NativeArray<int> Indices;
        public int NumPrimitives;
        
        public void Execute(int i)
        {
            if (i >= NumPrimitives) return;
            
            int originalIdx = Indices[i];
            edgeBVH edge = Edges[originalIdx];
            
            var node = Nodes[i];
            node.ObjectIndex = originalIdx;
            node.Min = Vector2.Min(edge.start, edge.end) - Vector2.one * 0.01f;
            node.Max = Vector2.Max(edge.start, edge.end) + Vector2.one * 0.01f;
            node.LeftChild = -1;
            node.RightChild = -1;
            Nodes[i] = node;
        }
    }
    
    [BurstCompile]
    unsafe struct ComputeInternalAABBJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<LBVHNodeRaw> Nodes;
        [NativeDisableParallelForRestriction]
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<int> AtomicCounters;
        public int NumPrimitives;
        
        public void Execute(int leafIdx)
        {
            if (leafIdx >= NumPrimitives) return;
            
            int currentIdx = Nodes[leafIdx].Parent;
            
            while (currentIdx != -1)
            {
                // 原子递增计数器，只有第二个到达的线程才处理
                int* counterPtr = (int*)AtomicCounters.GetUnsafePtr() + currentIdx;
                int count = System.Threading.Interlocked.Increment(ref *counterPtr);
                
                if (count < 2)
                {
                    // 第一个到达，等待另一个子节点
                    return;
                }
                
                // 第二个到达，计算 AABB
                var node = Nodes[currentIdx];
                int left = node.LeftChild;
                int right = node.RightChild;
                
                Vector2 minL = Nodes[left].Min;
                Vector2 maxL = Nodes[left].Max;
                Vector2 minR = Nodes[right].Min;
                Vector2 maxR = Nodes[right].Max;
                
                node.Min = Vector2.Min(minL, minR);
                node.Max = Vector2.Max(maxL, maxR);
                node.ObjectIndex = -1;
                Nodes[currentIdx] = node;
                
                currentIdx = node.Parent;
            }
        }
    }
    
    [BurstCompile]
    struct ReorderNodesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<LBVHNodeRaw> SourceNodes;
        [WriteOnly] public NativeArray<LBVHNodeRaw> DestNodes;
        [ReadOnly] public NativeArray<int> BfsQueue;
        [ReadOnly] public NativeArray<int> IndexMap;
        public int NodeCount;
        
        public void Execute(int i)
        {
            if (i >= NodeCount) return;
            
            int oldIdx = BfsQueue[i];
            var node = SourceNodes[oldIdx];
            
            if (node.LeftChild != -1)
                node.LeftChild = IndexMap[node.LeftChild];
            if (node.RightChild != -1)
                node.RightChild = IndexMap[node.RightChild];
            
            DestNodes[i] = node;
        }
    }
    
    [BurstCompile]
    struct PackGpuNodesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<LBVHNodeRaw> Nodes;
        [ReadOnly] public NativeArray<edgeBVH> Edges;
        [WriteOnly] public NativeArray<LBVHNodeGpu> GpuNodes;
        public int NodeCount;
        
        public void Execute(int i)
        {
            if (i >= NodeCount) return;
            
            var node = Nodes[i];
            LBVHNodeGpu gpuNode;
            
            if (node.ObjectIndex == -1)
            {
                // 内部节点
                gpuNode.PosA = node.Min;
                gpuNode.PosB = node.Max;
                gpuNode.IndexData = node.LeftChild;
                gpuNode.RightChild = node.RightChild;
            }
            else
            {
                // 叶子节点
                edgeBVH edge = Edges[node.ObjectIndex];
                gpuNode.PosA = edge.start;
                gpuNode.PosB = edge.end;
                gpuNode.IndexData = ~edge.matIdx;
                gpuNode.RightChild = 0;
            }
            
            GpuNodes[i] = gpuNode;
        }
    }
}
