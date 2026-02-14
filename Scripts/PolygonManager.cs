using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using Unity.Mathematics;
using UnityEditor.Experimental.GraphView;

namespace RadianceCascadesWorldBVH
{
    
    // bvh基础叶节点：一条边
    public struct edgeBVH
    {
        public Vector2 start;
        public Vector2 end;
        public int matIdx;
    }
    
    // BVH 节点结构
    public struct LBVHNode
    {
        public Vector2 Min;       // AABB 最小值
        public Vector2 Max;       // AABB 最大值
        public int LeftChild;     // 左孩子索引
        public int RightChild;    // 右孩子索引
        public int Parent;        // 父节点索引
        public int ObjectIndex;   //如果是叶子，这里存储原始边的索引(index数组的内容)；如果是内部节点，为-1
        public bool IsLeaf => LeftChild == -1; // 判断是否为叶子（约定叶子没有孩子）
    }

        
    // 材质数据结构
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct MaterialData
    {
        public Color BasicColor;       // 基础色
        [ColorUsage(false, true)] 
        public Color Emission;     // 发光
        public Vector4 uvBox;
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
        // 这么脚本目前完全以极低性能运行
        // 因为序列帧动画工作流没有接入
        // 要参与BVH的物体
        public List<SpriteRenderer> spriteRenderers;
        public List<RCWBObject> rcwObjects;
        
        // 场景包围盒
        public Vector4 sceneAABB;
        
        // 用于BVH的边集合（共享引用，需要传递给GPU）
        private List<edgeBVH> edges = new List<edgeBVH>();
        
        // BVH构建器
        private PolygonBVHConstructor bvhConstructor;
        
        // 材质数据（与rcwObjects完全同序）
        private List<MaterialData> materialData = new List<MaterialData>();
        
        // compute buffer
        private ComputeBuffer edgeBuffer;
        private ComputeBuffer nodeBuffer;
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
        
        void Start()
        {
            bvhConstructor = new PolygonBVHConstructor(edges, spriteRenderers);
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
                mat.uvBox = Vector4.zero; // TODO: 后续接入UV信息
                materialData.Add(mat);
            }
        }

        // Update is called once per frame
        void Update()
        {
            // 重新生成材质数据
            GenerateMaterialData();
            
            // 重新生成edges
            bvhConstructor.GetBvhEdges();
            // 计算莫顿码
            bvhConstructor.CalculateMortonCodes(sceneAABB);
            // 排序
            bvhConstructor.SortMortonCodes();
            // 生成BVH
            bvhConstructor.BuildBVHStructure();
            // 构建BVH的每个节点的包围盒
            bvhConstructor.RefitBVH();

            UpdateBuffers(edges, bvhConstructor.nodes);
        }
        
        
        public void UpdateBuffers(List<edgeBVH> edges, LBVHNode[] nodes)
        {
            // 1. 管理 Edge Buffer (叶子数据)
            // 如果edges不为零，那么要准备合适的edges
            if (edges.Count > 0)
            {
                // 如果 Buffer 不存在，或者长度不够，重新创建
                if (edgeBuffer == null || edgeBuffer.count < edges.Count)
                {
                    edgeBuffer?.Release();
                    // 建议：按 1.5 倍扩容，减少频繁分配
                    int newSize = Mathf.CeilToInt(edges.Count * 1.2f); 
                    int s = Marshal.SizeOf<edgeBVH>();
                    edgeBuffer = new ComputeBuffer(newSize, s);
                }
                // 注意：由于 List 可能有冗余容量，转换为 Array 传入
                edgeBuffer.SetData(edges.ToArray(), 0, 0, edges.Count);
            }

            // 2. 管理 Node Buffer (内部节点)
            if (nodes != null && nodes.Length > 0)
            {
                int nodeCount = nodes.Length;
                if (nodeBuffer == null || nodeBuffer.count < nodeCount)
                {
                    nodeBuffer?.Release();
                    int s = Marshal.SizeOf<LBVHNode>();
                    nodeBuffer = new ComputeBuffer(nodeCount, s); // 对应 NodeGPU 大小
                }
                Debug.Log("nodeBuffer set.");
                nodeBuffer.SetData(nodes, 0, 0, nodeCount);
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
            
            Shader.SetGlobalBuffer("_BVH_Edge_Buffer", edgeBuffer);
            Shader.SetGlobalBuffer("_BVH_Node_Buffer", nodeBuffer);
            Shader.SetGlobalBuffer("_BVH_Material_Buffer", materialBuffer);
            Shader.SetGlobalInt("_BVH_Root_Index", bvhConstructor.rootNodeIndex);
        }

        public void Dispose()
        {
            edgeBuffer?.Release();
            nodeBuffer?.Release();
            materialBuffer?.Release();
        }
        
        
        
        /// <summary>
        /// 递归绘制 BVH 节点
        /// </summary>
        /// <param name="nodeIndex">当前内部节点在数组中的索引</param>
        /// <param name="depth">当前深度</param>
        private void DrawNodeRecursive(int nodeIndex, int depth)
        {
            var bvhNodes = bvhConstructor.nodes;
            
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
            if (bvhConstructor != null && bvhConstructor.nodes != null)
            {
                var bvhNodes = bvhConstructor.nodes;
                
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
