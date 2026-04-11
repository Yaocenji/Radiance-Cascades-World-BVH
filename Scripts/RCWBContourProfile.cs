using System;
using System.Collections.Generic;
using UnityEngine;

namespace RadianceCascadesWorldBVH
{
    /// <summary>
    /// 单个轮廓环数据。
    /// pointsLocal 采用“对象局部空间”坐标，用于后续按对象 Transform 映射到世界空间。
    /// </summary>
    [Serializable]
    public class ContourLoopData
    {
        // 是否闭合环。闭合环会将最后一个点与第一个点自动连边。
        [SerializeField] private bool closed = true;
        // 轮廓顶点（局部空间）。建议按顺序存储，不应打乱点序。
        [SerializeField] private List<Vector2> pointsLocal = new List<Vector2>();

        /// <summary>是否为闭合环。</summary>
        public bool Closed => closed;
        /// <summary>局部空间点集（只读视图）。</summary>
        public IReadOnlyList<Vector2> PointsLocal => pointsLocal;
        /// <summary>点数量。</summary>
        public int PointCount => pointsLocal == null ? 0 : pointsLocal.Count;

        /// <summary>
        /// 基础合法性校验：
        /// 闭合环至少需要 3 个点；非闭合折线至少需要 2 个点。
        /// </summary>
        public bool IsValid()
        {
            if (pointsLocal == null) return false;
            return closed ? pointsLocal.Count >= 3 : pointsLocal.Count >= 2;
        }

        public void SetData(IReadOnlyList<Vector2> points, bool isClosed)
        {
            closed = isClosed;
            pointsLocal.Clear();

            if (points == null) return;
            for (int i = 0; i < points.Count; i++)
            {
                pointsLocal.Add(points[i]);
            }
        }
    }

    /// <summary>
    /// RCWB 对象级轮廓配置资产。
    /// 当前只负责“数据存储 + 基础校验 + 基础统计”，不直接参与 BVH 构建流程。
    /// </summary>
    [CreateAssetMenu(fileName = "RCWBContourProfile", menuName = "RadianceCascadesWorldBVH/RCWB Contour Profile")]
    public class RCWBContourProfile : ScriptableObject
    {
        // 一个 Profile 可以包含多个轮廓环（例如外轮廓 + 若干附加轮廓）。
        [SerializeField] private List<ContourLoopData> loops = new List<ContourLoopData>();

        /// <summary>轮廓环列表（只读视图）。</summary>
        public IReadOnlyList<ContourLoopData> Loops => loops;
        /// <summary>轮廓环数量。</summary>
        public int LoopCount => loops == null ? 0 : loops.Count;

        /// <summary>
        /// Profile 级合法性校验：
        /// 1) 至少存在 1 个环；
        /// 2) 每个环都非空且通过环级校验。
        /// </summary>
        public bool IsValid()
        {
            if (loops == null || loops.Count == 0) return false;

            for (int i = 0; i < loops.Count; i++)
            {
                ContourLoopData loop = loops[i];
                if (loop == null || !loop.IsValid())
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// 统计该 Profile 理论上的边数量。
        /// 闭合环：边数 = 点数；非闭合：边数 = 点数 - 1。
        /// </summary>
        public int GetEdgeCount()
        {
            if (loops == null || loops.Count == 0) return 0;

            int edgeCount = 0;
            for (int i = 0; i < loops.Count; i++)
            {
                ContourLoopData loop = loops[i];
                if (loop == null || !loop.IsValid()) continue;

                edgeCount += loop.Closed ? loop.PointCount : loop.PointCount - 1;
            }

            return edgeCount;
        }

        public void SetLoops(List<ContourLoopData> newLoops)
        {
            loops = newLoops ?? new List<ContourLoopData>();
        }
    }
}
