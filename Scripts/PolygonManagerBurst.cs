using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

// 该脚本是错误的，之后应该重写他们

namespace RadianceCascadesWorldBVH
{
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct EdgeGPU
    {
        public float2 start;
        public float2 end;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct NodeGPU
    {
        public float2 Min;          // 8 字节
        public float2 Max;          // 8 字节 (Min+Max = 16)
    
        public int LeftChild;       // 4 字节
        public int RightChild;      // 4 字节
        public int Parent;          // 4 字节
        public int ObjectIndex;     // 4 字节 (这一行 = 16)
    
        public int Padding1;        // 4 字节
        public int Padding2;        // 4 字节
        public int Padding3;        // 4 字节
        public int Padding4;        // 4 字节 (这一行 = 16)
    
        // 总计: 16 + 16 + 16 = 48 字节
    }

    public class PolygonManagerBurst : MonoBehaviour
    {
        public List<SpriteRenderer> spriteRenderers;
        public Vector4 sceneAABB;

        // 核心数据：高32位存莫顿码，低32位存原始索引
        private NativeArray<ulong> _sortData; 
        private NativeArray<EdgeGPU> _edges;
        private NativeArray<NodeGPU> _nodes;
        private NativeArray<int> _refitFlags;

        private ComputeBuffer _edgeBuffer;
        private ComputeBuffer _nodeBuffer;
        
        [Header("Gizmos Debug")]
        public bool showGizmos = true;
        public List<BVHDrawParam> depthColors = new List<BVHDrawParam>() 
        { 
            new BVHDrawParam(true, Color.white), 
            new BVHDrawParam(true, Color.red), 
            new BVHDrawParam(true, Color.green), 
            new BVHDrawParam(true, Color.blue),
            new BVHDrawParam(true, Color.yellow),
            new BVHDrawParam(true, Color.cyan),
            new BVHDrawParam(true, Color.magenta)
        };

        private void OnDisable()
        {
            DisposeBuffers();
        }

        private void DisposeBuffers()
        {
            if (_edges.IsCreated) _edges.Dispose();
            if (_sortData.IsCreated) _sortData.Dispose();
            if (_nodes.IsCreated) _nodes.Dispose();
            if (_refitFlags.IsCreated) _refitFlags.Dispose();

            _edgeBuffer?.Release();
            _nodeBuffer?.Release();
        }

        unsafe void Update()
        {
            List<EdgeGPU> tempEdges = GatherEdgesFromSprites();
            int n = tempEdges.Count;
            if (n < 2) return;

            ReallocateIfNeeded(n);
            _edges.CopyFrom(tempEdges.ToArray());

            float2 aabbMin = new float2(sceneAABB.x, sceneAABB.y);
            float2 aabbMax = new float2(sceneAABB.z, sceneAABB.w);

            // 1. 计算莫顿码并打包数据
            var mortonJob = new MortonCodePackJob
            {
                Edges = _edges,
                Min = aabbMin,
                Max = aabbMax,
                SortData = _sortData
            };
            JobHandle handle = mortonJob.Schedule(n, 64);

            // 2. 排序 (对单个 ulong 数组排序，无兼容性问题)
            var sortJob = new SortSingleArrayJob { Data = _sortData };
            handle = sortJob.Schedule(handle);

            // 3. 构建拓扑结构
            var buildJob = new BuildHierarchyJob
            {
                SortData = _sortData,
                Nodes = _nodes,
                NumLeaves = n
            };
            handle = buildJob.Schedule(n - 1, 32, handle);

            // 4. 重置 Refit 标记
            UnsafeUtility.MemSet(_refitFlags.GetUnsafePtr(), 0, _refitFlags.Length * sizeof(int));
            
            // 5. 自底向上更新包围盒
            var refitJob = new RefitAABBJob
            {
                Edges = _edges,
                SortData = _sortData,
                Nodes = _nodes,
                Flags = _refitFlags,
                NumLeaves = n
            };
            handle = refitJob.Schedule(n, 64, handle);

            handle.Complete();

            UploadToGPU(n);
        }

        private List<EdgeGPU> GatherEdgesFromSprites()
        {
            List<EdgeGPU> temp = new List<EdgeGPU>();
            foreach (var sr in spriteRenderers)
            {
                if (sr == null || sr.sprite == null) continue;
                int count = sr.sprite.GetPhysicsShapeCount();
                for (int i = 0; i < count; i++)
                {
                    List<Vector2> points = new List<Vector2>();
                    sr.sprite.GetPhysicsShape(i, points);
                    for (int j = 0; j < points.Count; j++)
                    {
                        Vector3 worldP1 = sr.transform.TransformPoint(points[j]);
                        Vector3 worldP2 = sr.transform.TransformPoint(points[(j + 1) % points.Count]);
                        temp.Add(new EdgeGPU { 
                            start = new float2(worldP1.x, worldP1.y), 
                            end = new float2(worldP2.x, worldP2.y) 
                        });
                    }
                }
            }
            return temp;
        }

        private unsafe void ReallocateIfNeeded(int n)
        {
            int nodeCount = 2 * n - 1;
            if (!_edges.IsCreated || _edges.Length < n)
            {
                DisposeBuffers();
                _edges = new NativeArray<EdgeGPU>(n, Allocator.Persistent);
                _sortData = new NativeArray<ulong>(n, Allocator.Persistent);
                _nodes = new NativeArray<NodeGPU>(nodeCount, Allocator.Persistent);
                _refitFlags = new NativeArray<int>(nodeCount, Allocator.Persistent);

                // 使用 sizeof 确保步长完全一致
                _edgeBuffer = new ComputeBuffer(n, UnsafeUtility.SizeOf<EdgeGPU>());
                _nodeBuffer = new ComputeBuffer(nodeCount, UnsafeUtility.SizeOf<NodeGPU>());
            }
        }

        private void UploadToGPU(int n)
        {
            _edgeBuffer.SetData(_edges, 0, 0, n);
            _nodeBuffer.SetData(_nodes, 0, 0, 2 * n - 1);
            Shader.SetGlobalBuffer("_EdgeBuffer", _edgeBuffer);
            Shader.SetGlobalBuffer("_NodeBuffer", _nodeBuffer);
            Shader.SetGlobalInt("_EdgeCount", n);
        }

        // --- JOBS ---

        [BurstCompile]
        struct MortonCodePackJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<EdgeGPU> Edges;
            public float2 Min;
            public float2 Max;
            public NativeArray<ulong> SortData;

            public void Execute(int i)
            {
                float2 center = (Edges[i].start + Edges[i].end) * 0.5f;
                float2 norm = math.saturate((center - Min) / (Max - Min));
                uint ix = (uint)(norm.x * 65535f);
                uint iy = (uint)(norm.y * 65535f);
                
                uint code = ExpandBits(ix) | (ExpandBits(iy) << 1);
                
                // 高32位莫顿码，低32位索引
                SortData[i] = ((ulong)code << 32) | (uint)i;
            }

            private uint ExpandBits(uint v)
            {
                v &= 0x0000FFFF;
                v = (v | (v << 8)) & 0x00FF00FFu;
                v = (v | (v << 4)) & 0x0F0F0F0Fu;
                v = (v | (v << 2)) & 0x33333333u;
                v = (v | (v << 1)) & 0x55555555u;
                return v;
            }
        }

        [BurstCompile]
        struct SortSingleArrayJob : IJob
        {
            public NativeArray<ulong> Data;
            public void Execute() => Data.Sort();
        }

        [BurstCompile]
        struct BuildHierarchyJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<ulong> SortData;
            [NativeDisableContainerSafetyRestriction] public NativeArray<NodeGPU> Nodes;
            public int NumLeaves;

            public void Execute(int i)
            {
                int d = (GetDelta(i, i + 1) - GetDelta(i, i - 1) >= 0) ? 1 : -1;
                int minDelta = GetDelta(i, i - d);
                int lMax = 2;
                while (GetDelta(i, i + lMax * d) > minDelta) lMax *= 2;
                int l = 0;
                for (int t = lMax / 2; t >= 1; t /= 2)
                    if (GetDelta(i, i + (l + t) * d) > minDelta) l += t;
                int j = i + l * d;
                int deltaNode = GetDelta(i, j);
                int split = math.min(i, j);
                int step = math.abs(i - j);
                do {
                    step = (step + 1) >> 1;
                    int newSplit = split + step;
                    if (newSplit < math.max(i, j))
                        if (GetDelta(math.min(i, j), newSplit) > deltaNode) split = newSplit;
                } while (step > 1);

                int currentNodeIdx = NumLeaves + i;
                int lChild = (math.min(i, j) == split) ? split : (NumLeaves + split);
                int rChild = (math.max(i, j) == split + 1) ? (split + 1) : (NumLeaves + split + 1);

                NodeGPU node = Nodes[currentNodeIdx];
                node.LeftChild = lChild;
                node.RightChild = rChild;
                node.Parent = -1; 
                Nodes[currentNodeIdx] = node;

                NodeGPU left = Nodes[lChild]; left.Parent = currentNodeIdx; Nodes[lChild] = left;
                NodeGPU right = Nodes[rChild]; right.Parent = currentNodeIdx; Nodes[rChild] = right;
            }

            int GetDelta(int i, int j)
            {
                if (j < 0 || j >= NumLeaves) return -1;
                uint codeI = (uint)(SortData[i] >> 32);
                uint codeJ = (uint)(SortData[j] >> 32);
                if (codeI != codeJ) return math.lzcnt(codeI ^ codeJ);
                // 莫顿码相同时，使用索引作为 tie-breaker
                return 32 + math.lzcnt((uint)SortData[i] ^ (uint)SortData[j]);
            }
        }

        [BurstCompile]
        struct RefitAABBJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<EdgeGPU> Edges;
            [ReadOnly] public NativeArray<ulong> SortData;
            [NativeDisableContainerSafetyRestriction] public NativeArray<NodeGPU> Nodes;
            [NativeDisableContainerSafetyRestriction] public NativeArray<int> Flags;
            public int NumLeaves;

            public unsafe void Execute(int i)
            {
                // 从打包数据中还原原始边索引
                int origIdx = (int)(SortData[i] & 0xFFFFFFFFu);
                
                NodeGPU node = Nodes[i];
                node.Min = math.min(Edges[origIdx].start, Edges[origIdx].end);
                node.Max = math.max(Edges[origIdx].start, Edges[origIdx].end);
                node.ObjectIndex = origIdx; // 存入 GPU
                node.LeftChild = -1;
                node.RightChild = -1;
                Nodes[i] = node;

                int current = node.Parent;
                int* flagsPtr = (int*)Flags.GetUnsafePtr();

                while (current != -1)
                {
                    int visited = System.Threading.Interlocked.Increment(ref flagsPtr[current]);
                    if (visited < 2) return; 

                    NodeGPU n = Nodes[current];
                    NodeGPU left = Nodes[n.LeftChild];
                    NodeGPU right = Nodes[n.RightChild];
                    n.Min = math.min(left.Min, right.Min);
                    n.Max = math.max(left.Max, right.Max);
                    Nodes[current] = n;

                    current = n.Parent;
                }
            }
        }
        
        
        
        
        
        
        private void OnDrawGizmos()
        {
            if (!showGizmos || !_nodes.IsCreated || _edges.Length < 2) return;

            int n = _edges.Length;
            int rootIndex = -1;

            // 1. 在内部节点区间 [n, 2n-2] 寻找根节点 (Parent == -1)
            for (int i = n; i < _nodes.Length; i++)
            {
                if (_nodes[i].Parent == -1)
                {
                    rootIndex = i;
                    break;
                }
            }

            // 2. 绘制场景 AABB
            Gizmos.color = Color.gray;
            Vector3 sceneCenter = new Vector3((sceneAABB.x + sceneAABB.z) * 0.5f, (sceneAABB.y + sceneAABB.w) * 0.5f, 0);
            Vector3 sceneSize = new Vector3(sceneAABB.z - sceneAABB.x, sceneAABB.w - sceneAABB.y, 0.1f);
            Gizmos.DrawWireCube(sceneCenter, sceneSize);

            // 3. 递归绘制 BVH 节点
            if (rootIndex != -1)
            {
                DrawNodeRecursive(rootIndex, 0, n);
            }
        }

        private void DrawNodeRecursive(int nodeIdx, int depth, int n)
        {
            if (nodeIdx < 0 || nodeIdx >= _nodes.Length) return;

            NodeGPU node = _nodes[nodeIdx];

            // 获取当前层级的绘制参数
            BVHDrawParam drawParam = depthColors[depth % depthColors.Count];
    
            if (drawParam.ifDraw)
            {
                Gizmos.color = drawParam.color;
                Vector3 center = new Vector3((node.Min.x + node.Max.x) * 0.5f, (node.Min.y + node.Max.y) * 0.5f, 0);
                Vector3 size = new Vector3(node.Max.x - node.Min.x, node.Max.y - node.Min.y, 0);
        
                // 稍微根据深度偏移一下 Z 轴，防止线段完全重叠产生的闪烁
                center.z = -depth * 0.01f; 
                Gizmos.DrawWireCube(center, size);
            }

            // 如果是内部节点 (索引 >= n)，继续向下递归
            if (nodeIdx >= n)
            {
                DrawNodeRecursive(node.LeftChild, depth + 1, n);
                DrawNodeRecursive(node.RightChild, depth + 1, n);
            }
        }
    }
}