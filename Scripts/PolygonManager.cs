using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using Unity.Mathematics;

#if UNITY_EDITOR
using UnityEditor.Experimental.GraphView;
#endif

namespace RadianceCascadesWorldBVH
{
    
    // bvh基础叶节点：一条边
    public struct edgeBVH
    {
        public Vector2 start;
        public Vector2 end;
        public int matIdx;
    }
    
    // BVH 节点结构 CPU端 未排序的
    public struct LBVHNodeRaw
    {
        public Vector2 Min;       // AABB 最小值
        public Vector2 Max;       // AABB 最大值
        public int LeftChild;     // 左孩子索引
        public int RightChild;    // 右孩子索引
        public int Parent;        // 父节点索引
        public int ObjectIndex;   //如果是叶子，这里存储原始边的索引(index数组的内容)；如果是内部节点，为-1
        public bool IsLeaf => LeftChild == -1; // 判断是否为叶子（约定叶子没有孩子）
    }
    
    // 定义一个用于上传的结构，与 Shader 严格对应
    /// <summary>
    /// LBVHNodeGpu
    /// PosA 如果不是叶节点，那么
    /// </summary>
    public struct LBVHNodeGpu
    {
        // 复用区域 1: 几何/空间信息 (16 bytes)
        // 内部BVH节点: AABB Min xy
        // 叶子BVH节点: Edge Start xy
        public Vector2 PosA;
        // 复用区域 2: 几何/空间信息 (16 bytes)
        // 内部BVH节点: AABB Max (xy)
        // 叶子BVH节点: Edge End (xy)
        public Vector2 PosB;
        // 复用区域 3: 索引/元数据 (4 bytes)
        // 内部BVH节点: Left Child Index (>= 0) 左子的索引
        // 叶子BVH节点: Bitwise NOT of Material Index (< 0) -> ~MatIdx 全部取反，作为材质的索引
        public int IndexData;
        // 复用区域 4: 辅助索引 (4 bytes)
        // 内部BVH节点: Right Child Index  右子的索引
        // 叶子BVH节点: Unused (or Padding) 暂无用
        public int RightChild;
    }
        
    // 材质数据结构
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct MaterialData
    {
        public Color BasicColor;       // 基础色
        [ColorUsage(false, true)] 
        public Color Emission;     // 发光
        public Vector4 uvMatrix;
        public Vector2 uvTranslation;
        public float Density;       // 物质密度
        public int TextureIndex;   // 如果你有多张图集，这里标记用哪张，单张图集可忽略
        private float _padding0;
        private float _padding1;
    }
    
    
    
    [Serializable]
    public struct BVHDrawParam
    {
        public bool ifDraw;
        public Color color;

        public BVHDrawParam(bool ifDraw, Color color)
        {
            this.ifDraw = ifDraw;
            this.color = color;
        }
    }
    
    public class PolygonManager : MonoBehaviour
    {
        // 静态实例，用于 RCWBObject 的注册/反注册
        public static PolygonManager Instance { get; private set; }
        
        [Header("Debug Settings")]
        // 每一层深度的颜色，如果深度超过列表长度，会循环使用
        public List<BVHDrawParam> depthColors = new List<BVHDrawParam>() 
        { 
            new BVHDrawParam(true, Color.white), 
            new BVHDrawParam(true, Color.white), 
            new BVHDrawParam(true, Color.white), 
            new BVHDrawParam(true, Color.white),
            new BVHDrawParam(true, Color.white),
            new BVHDrawParam(true, Color.white),
            new BVHDrawParam(true, Color.white)
        };
        
        [Header("BVH Settings")]
        // 要参与BVH的物体（通过注册机制自动管理）
        private List<SpriteRenderer> spriteRenderers = new List<SpriteRenderer>();
        private List<RCWBObject> rcwObjects = new List<RCWBObject>();
        
        // 场景包围盒
        public Vector4 sceneAABB;
        
        // 用于BVH的边集合（共享引用，需要传递给GPU）
        private List<edgeBVH> edges = new List<edgeBVH>();
        
        // BVH构建器
        private PolygonBVHConstructor bvhConstructor;
        private PolygonBVHConstructorAccelerated bvhConstructorAccelerated;
        
        // 材质数据（与rcwObjects完全同序）
        private List<MaterialData> materialData = new List<MaterialData>();
        
        // compute buffer
        private ComputeBuffer edgeBuffer;
        private ComputeBuffer nodeBuffer;
        private ComputeBuffer gpuNodeEdgeBuffer;
        private ComputeBuffer materialBuffer;
        
        // 辅助函数：计算多边形面积
        // 计算有向面积
        // > 0 : CCW (Counter-Clockwise)
        // < 0 : CW (Clockwise)
        public static float CalculateSignedArea(List<Vector2> points)
        {
            float area = 0f;
            for (int i = 0; i < points.Count; i++)
            {
                Vector2 p1 = points[i];
                Vector2 p2 = points[(i + 1) % points.Count]; // 循环回起点
                area += (p1.x * p2.y - p2.x * p1.y);
            }
            return area * 0.5f;
        }
        
        private void Awake()
        {
            // 设置单例实例
            if (Instance != null && Instance != this)
            {
                Debug.LogWarning("PolygonManager: 场景中存在多个实例，销毁重复的实例。");
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }
        
        void Start()
        {
            // 扫描场景中所有活跃的 RCWBObject，处理启动顺序导致的未注册情况
            ScanAndRegisterExistingObjects();
            
            bvhConstructor = new PolygonBVHConstructor(edges, spriteRenderers);
            bvhConstructorAccelerated = new PolygonBVHConstructorAccelerated(edges, spriteRenderers);
        }
        
        /// <summary>
        /// 扫描场景中所有活跃的 RCWBObject 并注册（用于处理启动顺序问题）
        /// </summary>
        private void ScanAndRegisterExistingObjects()
        {
            RCWBObject[] existingObjects = FindObjectsByType<RCWBObject>(FindObjectsSortMode.None);
            foreach (var obj in existingObjects)
            {
                if (obj.isActiveAndEnabled)
                {
                    SpriteRenderer sr = obj.GetComponent<SpriteRenderer>();
                    if (sr != null)
                    {
                        Register(obj, sr);
                    }
                }
            }
        }
        
        /// <summary>
        /// 注册一个 RCWBObject 及其对应的 SpriteRenderer
        /// </summary>
        /// <param name="rcwbObject">要注册的 RCWBObject</param>
        /// <param name="spriteRenderer">对应的 SpriteRenderer（必须挂载在同一物体上）</param>
        public void Register(RCWBObject rcwbObject, SpriteRenderer spriteRenderer)
        {
            if (rcwbObject == null || spriteRenderer == null)
            {
                Debug.LogWarning("PolygonManager.Register: rcwbObject 或 spriteRenderer 为空，跳过注册。");
                return;
            }
            
            // 检查是否已经注册
            if (rcwObjects.Contains(rcwbObject))
            {
                return;
            }
            
            // 同时添加到两个列表，保证一一对应
            rcwObjects.Add(rcwbObject);
            spriteRenderers.Add(spriteRenderer);
        }
        
        /// <summary>
        /// 反注册一个 RCWBObject
        /// </summary>
        /// <param name="rcwbObject">要反注册的 RCWBObject</param>
        public void Unregister(RCWBObject rcwbObject)
        {
            if (rcwbObject == null) return;
            
            int index = rcwObjects.IndexOf(rcwbObject);
            if (index >= 0)
            {
                // 同时从两个列表移除，保证一一对应
                rcwObjects.RemoveAt(index);
                spriteRenderers.RemoveAt(index);
            }
        }
        
        /// <summary>
        /// 从rcwObjects列表按完全相同的顺序生成材质数据列表
        /// </summary>
        private void GenerateMaterialData()
        {
            materialData.Clear();
            
            for (int i = 0; i < rcwObjects.Count; i++)
            {
                RCWBObject obj = rcwObjects[i];
                MaterialData mat = new MaterialData();
                mat.BasicColor = obj.BasicColor;
                mat.Density = obj.Density;
                mat.Emission = obj.Emission;
                mat.TextureIndex = 0; // TODO: 后续接入图集索引
                mat.uvMatrix = obj.UVMatrix;
                mat.uvTranslation = obj.UVTranslation;
                materialData.Add(mat);
            }
        }

        // Update is called once per frame
        void Update()
        {
            if (spriteRenderers.Count > 0)
            {
                BindAtlasGlobal(spriteRenderers[0].sprite);
            }
            
            // 重新生成材质数据
            GenerateMaterialData();
            
            /*// 重新生成edges
            bvhConstructor.GetBvhEdges();
            // 计算莫顿码
            bvhConstructor.CalculateMortonCodes(sceneAABB);
            // 排序
            bvhConstructor.SortMortonCodes();
            // 生成BVH
            bvhConstructor.BuildBVHStructure();
            // 构建BVH的每个节点的包围盒
            bvhConstructor.RefitBVH();
            // 重排BFS
            bvhConstructor.ReorderBVHToBFS();
            // 打包 GPU 数据
            bvhConstructor.PackGpuNodes();*/
            
            bvhConstructorAccelerated.GetBvhEdges();
            bvhConstructorAccelerated.CalculateMortonCodes(sceneAABB);
            bvhConstructorAccelerated.SortMortonCodes();
            bvhConstructorAccelerated.BuildBVHStructure();
            bvhConstructorAccelerated.RefitBVH();
            bvhConstructorAccelerated.ReorderBVHToBFS();
            bvhConstructorAccelerated.PackGpuNodes();

            //UpdateBuffers(edges, bvhConstructor.nodes);
            UpdateBuffers(edges, bvhConstructorAccelerated.nodes);
        }
        
        
        // 方法 A: 如果你手头有 Sprite 列表 (推荐用于你的 PolygonManager)
        public void BindAtlasGlobal(Sprite anySpriteInAtlas)
        {
            // 1. 获取颜色图集 (Albedo Atlas)
            // 运行时，这会自动指向打包后的大图
            Texture2D mainAtlas = anySpriteInAtlas.texture;

            // 2. 获取法线图集 (Normal Atlas)
            // 即使你没在 Inspector 里引用，Unity 也会生成平行图集
            //Texture2D normalAtlas = anySpriteInAtlas.GetSecondaryTexture("_NormalMap");

            // 3. 绑定到全局 Shader 变量
            // 这样你的 Compute Shader 和 片元着色器都能直接访问
            Shader.SetGlobalTexture("_RCWB_Atlas", mainAtlas);

            //Debug.Log($"Atlas Bound: {mainAtlas.name} ({mainAtlas.width}x{mainAtlas.height})");
        }
        
        
        public void UpdateBuffers(List<edgeBVH> edges, LBVHNodeRaw[] nodes)
        {
            // // 1. 管理 Edge Buffer (叶子数据)
            // // 如果edges不为零，那么要准备合适的edges
            // if (edges.Count > 0)
            // {
            //     // 如果 Buffer 不存在，或者长度不够，重新创建
            //     if (edgeBuffer == null || edgeBuffer.count < edges.Count)
            //     {
            //         edgeBuffer?.Release();
            //         // 建议：按 1.5 倍扩容，减少频繁分配
            //         int newSize = Mathf.CeilToInt(edges.Count * 1.2f); 
            //         int s = Marshal.SizeOf<edgeBVH>();
            //         edgeBuffer = new ComputeBuffer(newSize, s);
            //     }
            //     // 注意：由于 List 可能有冗余容量，转换为 Array 传入
            //     edgeBuffer.SetData(edges.ToArray(), 0, 0, edges.Count);
            // }

            // // 2. 管理 Node Buffer (内部节点)
            // if (nodes != null && nodes.Length > 0)
            // {
            //     int nodeCount = nodes.Length;
            //     if (nodeBuffer == null || nodeBuffer.count < nodeCount)
            //     {
            //         nodeBuffer?.Release();
            //         int s = Marshal.SizeOf<LBVHNodeRaw>();
            //         nodeBuffer = new ComputeBuffer(nodeCount, s); // 对应 NodeGPU 大小
            //     }
            //     //Debug.Log("nodeBuffer set.");
            //     nodeBuffer.SetData(nodes, 0, 0, nodeCount);
            // }
            
            // 2. 管理 GPU Node Edge Buffer (紧凑格式，内部节点+叶子边数据合并)
            int gpuNodeCount = bvhConstructorAccelerated.GpuNodeCount;// bvhConstructor.GpuNodeCount;
            if (gpuNodeCount > 0)
            {
                if (gpuNodeEdgeBuffer == null || gpuNodeEdgeBuffer.count < gpuNodeCount)
                {
                    gpuNodeEdgeBuffer?.Release();
                    int newSize = Mathf.NextPowerOfTwo(gpuNodeCount);
                    int s = Marshal.SizeOf<LBVHNodeGpu>();
                    gpuNodeEdgeBuffer = new ComputeBuffer(newSize, s);
                }
                gpuNodeEdgeBuffer.SetData(bvhConstructorAccelerated.gpuNodes/*bvhConstructor.gpuNodes*/, 0, 0, gpuNodeCount);
            }
            
            // 3. 管理 Material Buffer (材质数据)
            if (materialData.Count > 0)
            {
                if (materialBuffer == null || materialBuffer.count < materialData.Count)
                {
                    materialBuffer?.Release();
                    int newSize = Mathf.CeilToInt(materialData.Count * 1.2f);
                    int s = Marshal.SizeOf<MaterialData>();
                    materialBuffer = new ComputeBuffer(newSize, s);
                }
                materialBuffer.SetData(materialData.ToArray(), 0, 0, materialData.Count);
            }
            
            // Shader.SetGlobalBuffer("_BVH_Edge_Buffer", edgeBuffer);
            // Shader.SetGlobalBuffer("_BVH_Node_Buffer", nodeBuffer);
            Shader.SetGlobalBuffer("_BVH_NodeEdge_Buffer", gpuNodeEdgeBuffer);
            Shader.SetGlobalBuffer("_BVH_Material_Buffer", materialBuffer);
            Shader.SetGlobalInt("_BVH_Root_Index", bvhConstructorAccelerated/*bvhConstructor*/.rootNodeIndex);
        }

        public void OnDestroy()
        {
            edgeBuffer?.Release();
            nodeBuffer?.Release();
            gpuNodeEdgeBuffer?.Release();
            materialBuffer?.Release();
            bvhConstructorAccelerated?.Dispose();
            
            // 清理单例引用
            if (Instance == this)
            {
                Instance = null;
            }
        }
        
        
        
        /// <summary>
        /// 递归绘制 BVH 节点
        /// </summary>
        /// <param name="nodeIndex">当前内部节点在数组中的索引</param>
        /// <param name="depth">当前深度</param>
        private void DrawNodeRecursive(int nodeIndex, int depth)
        {
            var bvhNodes = bvhConstructorAccelerated/*bvhConstructor*/.nodes;
            
            // 越界保护
            if (nodeIndex < 0 || nodeIndex >= bvhNodes.Length) return;

            // 1. 设置颜色
            if (depthColors.Count > 0)
            {
                // 使用取模运算，如果深度超过颜色数量，则循环使用
                Gizmos.color = depthColors[depth % depthColors.Count].color;
            }
            else
            {
                Gizmos.color = Color.green; // 默认颜色
            }

            // 2. 获取节点数据
            var node = bvhNodes[nodeIndex];

            // 3. 绘制当前节点的 AABB
            // 注意：这是内部节点，它的 AABB 包裹了其下所有子节点
            Vector3 size = (Vector3)(node.Max - node.Min);
            Vector3 center = (Vector3)(node.Min + node.Max) * 0.5f;
        
            // 为了避免不同层级的线完全重叠看不清，可以根据深度稍微调整一下 Box 的 Z 轴大小或位置（可选）
            // 这里简单直接画
            if (depthColors[depth % depthColors.Count].ifDraw)
                Gizmos.DrawWireCube(center, size);

            // 4. 递归处理子节点
            // 如果子节点索引 >= 0，说明是内部节点，继续递归
            // 如果子节点索引 < 0，说明是叶子节点，通常叶子节点就是具体的边了，这里不再单独画叶子的框，
            // 因为上一层（当前层）的框已经刚好包裹住叶子了。
        
            if (node.LeftChild >= 0)
            {
                DrawNodeRecursive(node.LeftChild, depth + 1);
            }
        
            if (node.RightChild >= 0)
            {
                DrawNodeRecursive(node.RightChild, depth + 1);
            }
        }
        
        
        // 将边画出来
        private void OnDrawGizmos()
        {
            Gizmos.color = Color.white;
            // 画出包围盒
            Gizmos.DrawLine(new Vector3(sceneAABB.x, sceneAABB.y, 0f), new Vector3(sceneAABB.z, sceneAABB.y, 0f));
            Gizmos.DrawLine(new Vector3(sceneAABB.z, sceneAABB.y, 0f), new Vector3(sceneAABB.z, sceneAABB.w, 0f));
            Gizmos.DrawLine(new Vector3(sceneAABB.z, sceneAABB.w, 0f), new Vector3(sceneAABB.x, sceneAABB.w, 0f));
            Gizmos.DrawLine(new Vector3(sceneAABB.x, sceneAABB.w, 0f), new Vector3(sceneAABB.x, sceneAABB.y, 0f));
            
            // 画出所有的边的
            foreach (edgeBVH edge in edges)
            {
                Gizmos.color = materialData[edge.matIdx].BasicColor;
                Gizmos.DrawLine(edge.start, edge.end);
                //Gizmos.DrawSphere((edge.start + edge.end) / 2.0f, .1f);
            }
            
            // 画出莫顿码的z曲线
            /*for (int i = 0; i < edges.Count - 1; i++)
            {
                Vector2 point0 = (edges[index[i]].start + edges[index[i]].end) / 2.0f;
                Vector2 point1 = (edges[index[i + 1]].start + edges[index[i + 1]].end) / 2.0f;
                Gizmos.DrawLine(new Vector3(point0.x, point0.y, 0f), new Vector3(point1.x, point1.y, 0f));
            }*/
            
            
            // 画BVH
            if (/*bvhConstructor*/bvhConstructorAccelerated != null && /*bvhConstructor*/bvhConstructorAccelerated.nodes != null)
            {
                var bvhNodes = /*bvhConstructor*/bvhConstructorAccelerated.nodes;
                
                // 1. 寻找根节点 (Parent 为 -1 的节点)
                int rootIndex = -1;
                for (int i = 0; i < bvhNodes.Length; i++)
                {
                    if (bvhNodes[i].Parent == -1)
                    {
                        rootIndex = i;
                        break;
                    }
                }

                // 2. 从根节点开始递归绘制
                if (rootIndex != -1)
                {
                    DrawNodeRecursive(rootIndex, 0);
                }
            }
        }
    }
    
}
