using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class BooleanOperation
{
    /*public struct Edge
    {
        public Vector2 VertexA.Point;
        public Vector2 VertexB.Point;
    }*/

    // 边边求交系统
    public class IntersectionEdgeEdgeResult
    {
        public IntersectionEdgeEdgeResult()
        {
            intEdge = new Edge();
            intEdge.VertexBegin = new Vertex();
            intEdge.VertexEnd = new Vertex();
        }

        // 边的相交类型：单个点或重合一段
        public enum intersectionType
        {
            SinglePoint,
            SegmentOverlap
        }

        public intersectionType iType;

        // 是否平行
        public enum parellelType
        {
            Parallel,
            NonParallel,
        }

        public parellelType pType;

        // 单点相交的时候的特殊关系
        public enum singlePointSpecialRelationship
        {
            None,
            HeadTail, // 两条线段头尾相连交于一点
            TShape, // 两条线段成T字型交于一点
        }

        public singlePointSpecialRelationship spsr;

        // 头尾相接型，那么可能是A尾和B头重合，也有可能是B尾和A头重合，也可能是头头/尾尾
        public enum headTailType
        {
            HeadTail_AToB, // A尾连B头
            HeadTail_BToA, // B尾连A头
            HeadTail_ABHead, // AB头头连接（相背）
            HeadTail_ABTail, // AB尾尾连接（相向）
        }

        public headTailType hTType;

        // T字型交点，有四种情形
        public enum tShapeType
        {
            TShape_AinB,
            TShape_AoutB,
            TShape_BinA,
            TShape_BoutA,
        }

        public tShapeType tSType;


        // 如果是单个交点，那么这是交点坐标
        public Vector2 intPoint;

        // 如果是一段重合，那么这是这一段的前后坐标
        public Edge intEdge;

        // 记录：被求交的两条边
        public Edge aEdge;
        public Edge bEdge;
    }

    // 点和线段边的距离
    static public float DistancePointEdge(Edge edge, Vector2 point, ref Vector2 closestPoint)
    {
        // Vector from VertexA.Point to VertexB.Point
        Vector2 lineDirection = edge.VertexEnd.Point - edge.VertexBegin.Point;

        float lineLengthSqr = lineDirection.sqrMagnitude;

        if (lineLengthSqr == 0.0f)
        {
            return Vector2.Distance(point, edge.VertexBegin.Point);
        }

        float t = Vector2.Dot(point - edge.VertexBegin.Point, lineDirection) / lineLengthSqr;
        t = Mathf.Clamp01(t);

        closestPoint = edge.VertexBegin.Point + t * lineDirection;

        return Vector2.Distance(point, closestPoint);
    }

    // Helper function for 2D cross product
    public static float Cross(Vector2 a, Vector2 b)
    {
        return a.x * b.y - b.x * a.y;
    }

    public struct LoopInSameInfo
    {
        public Loop loopA;
        public Loop loopB;
    }
    //两条边是否同向一样或者反向一样
    public static bool TwoEdgesInSame(in Edge edge1, in Edge edge2)
    {
        return ((edge1.VertexBegin.Point - edge2.VertexBegin.Point).magnitude <= GeometryConstant.Tolerance &&
                (edge1.VertexEnd.Point - edge2.VertexEnd.Point).magnitude <= GeometryConstant.Tolerance) ||
               ((edge1.VertexBegin.Point - edge2.VertexEnd.Point).magnitude <= GeometryConstant.Tolerance &&
                (edge1.VertexBegin.Point - edge2.VertexEnd.Point).magnitude <= GeometryConstant.Tolerance);
    }
    // 两个Loop是否一样
    public static bool TwoLoopInSame(in Loop loop1, in Loop loop2)
    {
        bool ans = true;
        foreach (Edge edge1 in loop1.Edges)
        {
            bool tmpAns = false;
            foreach (Edge edge2 in loop2.Edges)
            {
                if (TwoEdgesInSame(edge1, edge2))
                {
                    tmpAns = true;
                    break;
                }
            }
            if (!tmpAns)
            {
                ans = false;
                break;
            }
        }
        return ans;
    }

    // 线段边求交
    static public void IntersectionEdgeEdge(Edge edgeA, Edge edgeB, out IntersectionEdgeEdgeResult intersectionResult)
    {
        // By default, there is no intersection.
        intersectionResult = null;

        // Represent the edges as parametric equations: P = P_start + t * direction
        Vector2 p = edgeA.VertexBegin.Point;
        Vector2 r = edgeA.VertexEnd.Point - edgeA.VertexBegin.Point;
        Vector2 q = edgeB.VertexBegin.Point;
        Vector2 s = edgeB.VertexEnd.Point - edgeB.VertexBegin.Point;

        // Calculate the 2D cross product of the direction vectors
        float r_cross_s = r.x * s.y - r.y * s.x;
        Vector2 q_minus_p = q - p;
        float q_minus_p_cross_r = q_minus_p.x * r.y - q_minus_p.y * r.x;

        // Case 1: The lines are parallel or collinear
        if (Mathf.Abs(r_cross_s) < GeometryConstant.Tolerance)
        {
            // Check if they are also collinear. If not, they are parallel and non-intersecting.
            if (Mathf.Abs(q_minus_p_cross_r) > GeometryConstant.Tolerance)
            {
                return; // Parallel and non-intersecting
            }

            // --- Lines are collinear, now check for segment overlap ---
            float r_dot_r = Vector2.Dot(r, r);

            // Handle case where edgeA is just a point
            if (r_dot_r < GeometryConstant.Tolerance)
            {
                // Check if this point lies on edgeB
                float s_dot_s = Vector2.Dot(s, s);
                if (s_dot_s < GeometryConstant.Tolerance) // Both are points
                {
                    if (Vector2.Distance(p, q) < GeometryConstant.Tolerance)
                    {
                        // Two points at the same location
                        intersectionResult = new IntersectionEdgeEdgeResult();
                        intersectionResult.pType = IntersectionEdgeEdgeResult.parellelType.Parallel;
                        intersectionResult.iType = IntersectionEdgeEdgeResult.intersectionType.SinglePoint;
                        intersectionResult.spsr = IntersectionEdgeEdgeResult.singlePointSpecialRelationship.HeadTail;
                        intersectionResult.hTType =
                            IntersectionEdgeEdgeResult.headTailType.HeadTail_ABHead; // Or any, it's ambiguous
                        intersectionResult.intPoint = p;
                    }

                    return;
                }

                float tmpu = Vector2.Dot(p - q, s) / s_dot_s;
                if (tmpu >= -GeometryConstant.Tolerance && tmpu <= 1 + GeometryConstant.Tolerance)
                {
                    // Point A lies on segment B
                    intersectionResult = new IntersectionEdgeEdgeResult();
                    intersectionResult.pType = IntersectionEdgeEdgeResult.parellelType.Parallel;
                    intersectionResult.iType = IntersectionEdgeEdgeResult.intersectionType.SinglePoint;
                    intersectionResult.intPoint = p;
                    intersectionResult.spsr = IntersectionEdgeEdgeResult.singlePointSpecialRelationship.TShape;
                    intersectionResult.tSType = IntersectionEdgeEdgeResult.tShapeType.TShape_AinB;
                }

                return;
            }

            float t0 = Vector2.Dot(q - p, r) / r_dot_r;
            float t1 = t0 + Vector2.Dot(s, r) / r_dot_r;

            float overlap_start = Mathf.Max(0.0f, Mathf.Min(t0, t1));
            float overlap_end = Mathf.Min(1.0f, Mathf.Max(t0, t1));

            // If the overlap interval is not valid, they are collinear but don't overlap
            if (overlap_start > overlap_end + GeometryConstant.Tolerance)
            {
                return;
            }

            // An intersection exists, so create the result object
            intersectionResult = new IntersectionEdgeEdgeResult();
            intersectionResult.aEdge = edgeA;
            intersectionResult.bEdge = edgeB;
            intersectionResult.pType = IntersectionEdgeEdgeResult.parellelType.Parallel;

            // Check if the overlap is just a single point
            if (Mathf.Abs(overlap_start - overlap_end) < GeometryConstant.Tolerance)
            {
                intersectionResult.iType = IntersectionEdgeEdgeResult.intersectionType.SinglePoint;
                intersectionResult.intPoint = p + overlap_start * r;

                // This is a collinear head-to-tail connection
                intersectionResult.spsr = IntersectionEdgeEdgeResult.singlePointSpecialRelationship.HeadTail;

                bool t0_is_0 = Mathf.Abs(t0) < GeometryConstant.Tolerance;
                bool t0_is_1 = Mathf.Abs(t0 - 1.0f) < GeometryConstant.Tolerance;
                bool t1_is_0 = Mathf.Abs(t1) < GeometryConstant.Tolerance;
                bool t1_is_1 = Mathf.Abs(t1 - 1.0f) < GeometryConstant.Tolerance;

                if (t0_is_1)
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_AToB; // A_end touches B_start
                else if (t1_is_0)
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_BToA; // B_end touches A_start
                else if (t0_is_0)
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_ABHead; // A_start touches B_start
                else if (t1_is_1)
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_ABTail; // A_end touches B_end
            }
            else // The overlap is a line segment
            {
                intersectionResult.iType = IntersectionEdgeEdgeResult.intersectionType.SegmentOverlap;
                intersectionResult.intEdge.VertexBegin.Point = p + overlap_start * r;
                intersectionResult.intEdge.VertexEnd.Point = p + overlap_end * r;
            }

            return;
        }

        // Case 2: Lines are not parallel and intersect at a single point
        // Solve for parameters t and u: p + t*r = q + u*s
        float t = (q_minus_p.x * s.y - q_minus_p.y * s.x) / r_cross_s;
        float u = q_minus_p_cross_r / r_cross_s;

        // Check if the intersection point lies on both segments (t and u are between 0 and 1)
        if (t >= -GeometryConstant.Tolerance && t <= 1 + GeometryConstant.Tolerance &&
            u >= -GeometryConstant.Tolerance && u <= 1 + GeometryConstant.Tolerance)
        {
            // An intersection exists, create and populate the result object
            intersectionResult = new IntersectionEdgeEdgeResult();
            intersectionResult.aEdge = edgeA;
            intersectionResult.bEdge = edgeB;
            intersectionResult.pType = IntersectionEdgeEdgeResult.parellelType.NonParallel;
            intersectionResult.iType = IntersectionEdgeEdgeResult.intersectionType.SinglePoint;
            intersectionResult.intPoint = p + t * r;

            // --- Now, classify the single point intersection ---
            bool a_is_headpoint = Mathf.Abs(t) < GeometryConstant.Tolerance;
            bool a_is_tailpoint = Mathf.Abs(t - 1.0f) < GeometryConstant.Tolerance;
            bool b_is_headpoint = Mathf.Abs(u) < GeometryConstant.Tolerance;
            bool b_is_tailpoint = Mathf.Abs(u - 1.0f) < GeometryConstant.Tolerance;

            bool a_is_endpoint = a_is_headpoint || a_is_tailpoint;
            bool b_is_endpoint = b_is_headpoint || b_is_tailpoint;

            if (a_is_endpoint && b_is_endpoint)
            {
                intersectionResult.spsr = IntersectionEdgeEdgeResult.singlePointSpecialRelationship.HeadTail;

                bool a_is_head = Mathf.Abs(t) < GeometryConstant.Tolerance;
                bool b_is_head = Mathf.Abs(u) < GeometryConstant.Tolerance;

                if (!a_is_head && b_is_head)
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_AToB; // A-tail to B-head
                else if (a_is_head && !b_is_head)
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_BToA; // B-tail to A-head
                else if (a_is_head && b_is_head)
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_ABHead; // Heads connect
                else
                    intersectionResult.hTType =
                        IntersectionEdgeEdgeResult.headTailType.HeadTail_ABTail; // Tails connect
            }
            else if (a_is_endpoint && !b_is_endpoint)
            {
                intersectionResult.spsr = IntersectionEdgeEdgeResult.singlePointSpecialRelationship.TShape;
                if (a_is_tailpoint)
                    intersectionResult.tSType =
                        IntersectionEdgeEdgeResult.tShapeType.TShape_AinB; // Tail of A is on segment B
                else if (a_is_headpoint)
                    intersectionResult.tSType =
                        IntersectionEdgeEdgeResult.tShapeType.TShape_AoutB; // Head of A is on segment B
            }
            else if (!a_is_endpoint && b_is_endpoint)
            {
                intersectionResult.spsr = IntersectionEdgeEdgeResult.singlePointSpecialRelationship.TShape;
                if (b_is_tailpoint)
                    intersectionResult.tSType =
                        IntersectionEdgeEdgeResult.tShapeType.TShape_BinA; // Tail of B is on segment A
                else if (b_is_headpoint)
                    intersectionResult.tSType =
                        IntersectionEdgeEdgeResult.tShapeType.TShape_BoutA; // Head of B is on segment A
            }
            else
            {
                // Standard "X" intersection where the point is not an endpoint for either segment
                intersectionResult.spsr = IntersectionEdgeEdgeResult.singlePointSpecialRelationship.None;
            }
        }
        // If the intersection point is not on both segments, intersectionResult remains null
    }

    // 入出点信息
    public class InOutPoint
    {
        // 点的位置
        public Vector2 Point;

        // 所连的A的边，可能有两个，那么按照前后的顺序放
        public Edge[] EdgeA;

        // 所连的B的边，可能有两个，那么按照前后的顺序放
        public Edge[] EdgeB;

        public InOutPoint()
        {
            EdgeA = new Edge[2];
            EdgeB = new Edge[2];
        }
    }

    static public void getAllEdgeIntersectionOfTwoMesh(ref Mesh meshA, ref Mesh meshB,
        ref List<IntersectionEdgeEdgeResult> results)
    {
        // 首先，拿到两个模型所有的边
        List<Edge> aEdges = new List<Edge>();
        List<Edge> bEdges = new List<Edge>();

        foreach (var loop in meshA.Loops)
        {
            aEdges.AddRange(loop.Edges);
        }

        foreach (var loop in meshB.Loops)
        {
            bEdges.AddRange(loop.Edges);
        }

        foreach (var aEdge in aEdges)
        {
            foreach (var bEdge in bEdges)
            {
                IntersectionEdgeEdgeResult currInt = null;
                IntersectionEdgeEdge(aEdge, bEdge, out currInt);

                // 真有相交情况
                if (currInt is not null)
                {
                    results.Add(currInt);
                }
            }
        }
    }


    // 根据相交点集合，计算出点和入点
    static public void CalculateInoutPoint(ref List<IntersectionEdgeEdgeResult> intersectionResults,
        ref List<InOutPoint> inPoints, ref List<InOutPoint> outPoints, bool IOA)
    {
        // 开始处理所有的求交结果
        // 第一步：放弃掉所有的重合类
        List<IntersectionEdgeEdgeResult> deletes = new List<IntersectionEdgeEdgeResult>();
        foreach (var result in intersectionResults)
        {
            if (result.iType == IntersectionEdgeEdgeResult.intersectionType.SegmentOverlap)
                deletes.Add(result);
        }

        foreach (var result in deletes)
        {
            intersectionResults.Remove(result);
        }

        // 处理过程中的标记
        bool[] visit = new bool[intersectionResults.Count];
        for (int i = 0; i < visit.Length; i++)
        {
            visit[i] = false;
        }

        // 将所有的相交信息转化为出、入点
        for (int i = 0; i < intersectionResults.Count; i++)
        {
            // 看过的就不看了
            if (visit[i])
                continue;

            var result = intersectionResults[i];
            
            // 得找到其他所有的和这个点同位置的交点
            List<IntersectionEdgeEdgeResult> samePointInt = new List<IntersectionEdgeEdgeResult>();
            samePointInt.Add(result);
            // 打上标记
            visit[i] = true;
            // 找到其他所有的和这个点同位置的交点
            for (int j = 0; j < intersectionResults.Count; j++)
            {
                if (visit[j])
                    continue;

                if (intersectionResults[j].iType == IntersectionEdgeEdgeResult.intersectionType.SinglePoint &&
                    (intersectionResults[j].intPoint - result.intPoint).magnitude <= GeometryConstant.Tolerance)
                {
                    // 打上标记
                    visit[j] = true;
                    // 记录这个同位置交点
                    samePointInt.Add(intersectionResults[j]);
                }
            }
            
            
            // 最普通的X型交点
            if (result.spsr == IntersectionEdgeEdgeResult.singlePointSpecialRelationship.None)
            {
                // 单点相交，那么判断B边的起点在A边的内外（左右）
                // 用右手cross，A边向量和A尾-B头向量 叉乘，若值为负，则可以说这是入点，否则是出点
                float inoutCriterion = Cross(result.aEdge.getVector, result.bEdge.getVector);

                InOutPoint newInoutPoint = new InOutPoint();
                newInoutPoint.Point = result.intPoint;
                newInoutPoint.EdgeA[0] = result.aEdge;
                newInoutPoint.EdgeB[0] = result.bEdge;

                if (inoutCriterion > 0)
                {
                    // 不论求交并，这个点是个入点
                    inPoints.Add(newInoutPoint);
                }
                else
                {
                    // 不论求交并，否则是个出点
                    outPoints.Add(newInoutPoint);
                }

                // 好，打上标记
                visit[i] = true;


                continue;
            }
            // T字型相交
            else if (result.spsr == IntersectionEdgeEdgeResult.singlePointSpecialRelationship.TShape)
            {
                InOutPoint newInoutPoint = new InOutPoint();
                newInoutPoint.Point = result.intPoint;
                newInoutPoint.EdgeA[0] = result.aEdge;
                newInoutPoint.EdgeB[0] = result.bEdge;

                // 根据当前重合交点的信息 判定入点、出点、放弃

                // 第一种，BinA然后BoutA，即B边来到A边，又离开A边。
                // 1、B的边恰在顶点处穿过了A的边；
                // 2、B的边在A的边的一侧弹跳一次；
                // 判断并且计算是哪种情况，如果是真正穿过了，那么判断是入点还是出点
                if (samePointInt.Count == 2 &&
                    ((samePointInt[0].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_BinA &&
                      samePointInt[1].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_BoutA) ||
                     (samePointInt[0].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_BoutA &&
                      samePointInt[1].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_BinA)))
                {
                    // 拿到In和Out对应的result
                    var inResult = samePointInt[0];
                    var outResult = samePointInt[1];
                    if (samePointInt[0].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_BoutA)
                    {
                        inResult = samePointInt[1];
                        outResult = samePointInt[0];
                    }

                    // 剔除一个理论上不会发生的错误：两个result的A边不是同一条
                    if (inResult.aEdge != outResult.aEdge)
                        continue; // 直接跑路

                    // 判断是真正穿过了还是在弹跳
                    // 真正穿过：inResult的b边起点 和 outResult的b边终点 分列在公共A边的两旁
                    // 用两个叉乘判据计算
                    float critIn = Cross(result.aEdge.getVector,
                        inResult.bEdge.VertexBegin.Point - result.aEdge.VertexBegin.Point);
                    float critOut = Cross(result.aEdge.getVector,
                        outResult.bEdge.VertexEnd.Point - result.aEdge.VertexBegin.Point);
                    // 两个叉乘结果同号，那么说明在同一边了，那么说明这个是一个“假点”，b边从a边一旁弹跳而过
                    if (critIn * critOut > 0)
                        continue; // 直接跑路
                    // 否则就是“真点”，b的边真正穿过了a的边

                    // 判断是出还是入
                    float inoutCriterion = Cross(result.aEdge.getVector, result.bEdge.getVector);

                    if (inoutCriterion > 0)
                    {
                        // 这是一个入点
                        inPoints.Add(newInoutPoint);
                    }
                    else
                    {
                        // 这是一个出点
                        outPoints.Add(newInoutPoint);
                    }
                }

                // 第二种，AinB然后AoutB，即A边来到B边，又离开B边。
                // 1、A的边恰在顶点处穿过了B的边；
                // 2、A的边在B的边的一侧弹跳一次；
                // 判断并且计算是哪种情况，如果是真正穿过了，那么判断是入点还是出点
                else if (samePointInt.Count == 2 &&
                         ((samePointInt[0].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_AinB &&
                           samePointInt[1].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_AoutB) ||
                          (samePointInt[0].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_AoutB &&
                           samePointInt[1].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_AinB)))
                {
                    // 拿到In和Out对应的result
                    var inResult = samePointInt[0];
                    var outResult = samePointInt[1];
                    if (samePointInt[0].tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_AoutB)
                    {
                        inResult = samePointInt[1];
                        outResult = samePointInt[0];
                    }

                    // 剔除一个理论上不会发生的错误：两个result的B边不是同一条
                    if (inResult.bEdge != outResult.bEdge)
                        continue; // 直接跑路

                    // 判断是真正穿过了还是在弹跳
                    // 真正穿过：inResult的a边起点 和 outResult的a边终点 分列在公共B边的两旁
                    // 用两个叉乘判据计算
                    float critIn = Cross(result.bEdge.getVector,
                        inResult.aEdge.VertexBegin.Point - result.bEdge.VertexBegin.Point);
                    float critOut = Cross(result.bEdge.getVector,
                        outResult.aEdge.VertexEnd.Point - result.bEdge.VertexBegin.Point);
                    // 两个叉乘结果同号，那么说明在同一边了，那么说明这个是一个“假点”，b边从a边一旁弹跳而过
                    if (critIn * critOut > 0)
                        continue; // 直接跑路
                    // 否则就是“真点”，a的边真正穿过了b的边

                    // 判断是出还是入
                    float inoutCriterion = Cross(result.aEdge.getVector, result.bEdge.getVector);

                    if (inoutCriterion > 0)
                    {
                        // 不论求交并，这是一个入点
                        inPoints.Add(newInoutPoint);
                    }
                    else
                    {
                        // 不论求交并，这是一个出点
                        outPoints.Add(newInoutPoint);
                    }
                }

                // 第三种，这个Tshape交点，没有与之重合位置的同位交点，说明与之相连的是：重合边
                else if (samePointInt.Count == 1)
                {
                    if (result.tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_BinA)
                    {
                        // 直接判断是 “疑似”出点 或  “疑似”入点
                        float inoutCriterion = Cross(result.aEdge.getVector, result.bEdge.getVector);
                        if (inoutCriterion > 0)
                        {
                            if (IOA)
                            {
                                // 疑似入点，相当于入了但没入，而是并到重合边去了，在求交算法中，当做没入
                                // inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                inPoints.Add(newInoutPoint);
                            }
                        }
                        else
                        {
                            if (IOA)
                            {
                                // 疑似出点，相当于出了但没出，而是并到重合边去了，在求交算法中，当做出了
                                outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // outPoints.Add(newInoutPoint);
                            }
                        }
                    }
                    else if (result.tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_BoutA)
                    {
                        // 直接判断是 “疑似”出点 或  “疑似”入点
                        float inoutCriterion = Cross(result.aEdge.getVector, result.bEdge.getVector);
                        if (inoutCriterion > 0)
                        {
                            if (IOA)
                            {
                                // 疑似入点，相当于入了但没入，而是并到重合边去了，在求交算法中，当做真入了
                                inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                //inPoints.Add(newInoutPoint);
                            }
                        }
                        else
                        {
                            if (IOA)
                            {
                                // 疑似出点，相当于出了但没出，而是并到重合边去了，在求交算法中，当做没出
                                // outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                outPoints.Add(newInoutPoint);
                            }
                        }
                    }
                    else if (result.tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_AinB)
                    {
                        float inoutCriterion = Cross(result.aEdge.getVector, result.bEdge.getVector);
                        // 判断是 “疑似”出点 或  “疑似”入点
                        // a插进b的情况，不根据当前的A边，而要看a的后续边和b的关系
                        // 判断：a的后续边是顺着b同方向走还是逆着反方向走
                        float inoutCriterionNext = Vector2.Dot(result.aEdge.VertexEnd.EdgeNext.getVector,
                            result.bEdge.getVector);

                        // 疑似入点，且a后续边和b同方向走
                        if (inoutCriterion > 0 && inoutCriterionNext > 0)
                        {
                            if (IOA)
                            {
                                // 在求交的算法中，当做一个“伪”入点
                                // inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 在求并算法中，当做一个真入点
                                inPoints.Add(newInoutPoint);
                            }
                        }
                        // 疑似入点，且a后续边和b反方向走
                        else if (inoutCriterion > 0 && inoutCriterionNext <= 0)
                        {
                            if (IOA)
                            {
                                // 在求交的算法中，当做一个“真”入点
                                inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 在求并算法中，当做一个伪入点
                                // inPoints.Add(newInoutPoint);
                            }
                        }
                        // 疑似出点，且a后续边和b同方向走
                        else if (inoutCriterion <= 0 && inoutCriterionNext > 0)
                        {
                            if (IOA)
                            {
                                // 在求交的算法中，当做一个真出点
                                outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 在求并算法中，当做一个伪出点
                                // outPoints.Add(newInoutPoint);
                            }
                        }
                        // 疑似出点，且a后续边和b反方向走
                        else // if (inoutCriterion <= 0 && inoutCriterionNext <= 0)
                        {
                            if (IOA)
                            {
                                // 在求交的算法中，当做一个“伪”出点
                                // outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 在求并算法中，当做一个真出点
                                outPoints.Add(newInoutPoint);
                            }
                        }
                    }
                    else if (result.tSType == IntersectionEdgeEdgeResult.tShapeType.TShape_AoutB)
                    {
                        float inoutCriterion = Cross(result.aEdge.getVector, result.bEdge.getVector);

                        // 判断是 “疑似”出点 或  “疑似”入点
                        // a从b分出的情况，不根据当前的A边，而要看a的前驱边和b的关系
                        // 判断：a的前驱边是顺着b同方向走还是逆着反方向走

                        float inoutCriterionNext = Vector2.Dot(result.aEdge.VertexBegin.EdgeLast.getVector,
                            result.bEdge.getVector);

                        // 疑似入点，且a前驱边和b同方向走
                        if (inoutCriterion > 0 && inoutCriterionNext > 0)
                        {
                            if (IOA){
                                // 在求交的算法中，当做一个“真”入点
                                inPoints.Add(newInoutPoint);
                            }
                            else{
                                // 在求并的算法中，当做一个“伪”入点
                                //inPoints.Add(newInoutPoint);
                            }
                        }
                        // 疑似入点，且a前驱边和b反方向走
                        else if (inoutCriterion > 0 && inoutCriterionNext <= 0)
                        {
                            if (IOA)
                            {
                                // 在求交的算法中，当做一个“伪”入点
                                // inPoints.Add(newInoutPoint);
                            }
                            else{
                                // 在求并的算法中，当做一个“真”入点
                                inPoints.Add(newInoutPoint);
                            }
                        }
                        // 疑似出点，且a前驱边和b同方向走
                        else if (inoutCriterion <= 0 && inoutCriterionNext > 0)
                        {
                            if (IOA)
                            {
                                // 在求交的算法中，当做一个“伪”出点
                                // outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 在求并的算法中，当做一个“真”出点
                                outPoints.Add(newInoutPoint);
                            }
                        }
                        // 疑似出点，且a前驱边和b反方向走
                        else // if (inoutCriterion <= 0 && inoutCriterionNext <= 0)
                        {
                            if (IOA){
                                // 在求交的算法中，当做一个“真”出点
                                outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 在求并的算法中，当做一个“伪”出点
                                // outPoints.Add(newInoutPoint);
                            }
                        }
                    }
                }

                continue;
            }
            // 头尾型相交
            else if (result.spsr == IntersectionEdgeEdgeResult.singlePointSpecialRelationship.HeadTail)
            {
                InOutPoint newInoutPoint = new InOutPoint();
                newInoutPoint.Point = result.intPoint;
                newInoutPoint.EdgeA[0] = result.aEdge;
                newInoutPoint.EdgeB[0] = result.bEdge;
                
                // 先找到这个十字路口 四个 ab边 的前后顺序
                int aIdx = 0, bIdx = 0;
                Edge[] aEdges = new Edge[]{null, null};
                Edge[] bEdges = new Edge[]{null, null};
                
                Edge aEdgePre = null, aEdgeNxt = null;
                Edge bEdgePre = null, bEdgeNxt = null;

                // 先拿到四个边
                foreach (var currResult in samePointInt)
                {
                    if (aIdx == 0)
                    {
                        aEdges[0] = currResult.aEdge;
                        aIdx++;
                    }
                    else if (aIdx == 1)
                    {
                        if (aEdges[0] != currResult.aEdge)
                        {
                            aEdges[1] = currResult.aEdge;
                            aIdx++;
                        }
                    }
                    else
                    {
                        // do nothing,
                    }
                    
                    if (bIdx == 0)
                    {
                        bEdges[0] = currResult.bEdge;
                        bIdx++;
                    }
                    else if (bIdx == 1)
                    {
                        if (bEdges[0] != currResult.bEdge)
                        {
                            bEdges[1] = currResult.bEdge;
                            bIdx++;
                        }
                    }
                    else
                    {
                        // do nothing,
                    }
                }
                
                // 然后a、b边组分别地排序
                // a[0]在前，a[1]在后
                if ((aEdges[0].VertexEnd.Point - aEdges[1].VertexBegin.Point).magnitude <=
                    GeometryConstant.Tolerance)
                {
                    aEdgePre = aEdges[0];
                    aEdgeNxt = aEdges[1];
                }
                // a[1]在前，a[0]在后
                else
                {
                    aEdgePre = aEdges[1];
                    aEdgeNxt = aEdges[0];
                }
                
                // b[0]在前，b[1]在后
                if ((bEdges[0].VertexEnd.Point - bEdges[1].VertexBegin.Point).magnitude <=
                    GeometryConstant.Tolerance)
                {
                    bEdgePre = bEdges[0];
                    bEdgeNxt = bEdges[1];
                }
                // b[1]在前，b[0]在后
                else
                {
                    bEdgePre = bEdges[1];
                    bEdgeNxt = bEdges[0];
                }

                // 情况一，同位交点有4个，说明是十字路口型
                if (samePointInt.Count == 4)
                {
                    // 判断这个十字路口是 ab互穿 还是 弹射擦过
                    
                    
                    // 1、计算A的两个向量和 sumA
                    // 2、判断 bEdgePre 的的头和 bEdgeNxt 的尾，在sumA的两侧还是一侧
                    // 3、如果在两侧，那么就是ab互穿，如果在一侧，那就是弹射而过
                    
                    Vector2 aSumVector = aEdgePre.getVector + aEdgeNxt.getVector;
                    Vector2 bSumVector = bEdgePre.getVector + bEdgeNxt.getVector;
                    Vector2 aHead2BpreBegin = bEdgePre.VertexBegin.Point - aEdgePre.VertexBegin.Point;
                    Vector2 aHead2BnxtEnd = bEdgeNxt.VertexEnd.Point - aEdgePre.VertexBegin.Point;
                    
                    // 两个叉乘 判断其同号或异号
                    float crit0 = Cross(aSumVector, aHead2BpreBegin) * Cross(aSumVector, aHead2BnxtEnd);
                    
                    // 总之，拿到了十字路口的四个边，也拿到了他们的前后关系
                    // 判断这个十字路口是 ab互穿 还是 弹射擦过
                    // 考虑三组向量之间的逆时针夹角:
                    // -aEdgePre x aEdgeNxt
                    // -aEdgePre x -bEdgePre
                    // -aEdgePre x bEdgeNxt
                    // 如果-aEdgePre x aEdgeNxt 大于另外两者或小于另外两者，那么弹射而过
                    // 否则是ab互穿
                    
                    float angleCriterion0 = Vector2.SignedAngle(-aEdgePre.getVector.normalized, aEdgeNxt.getVector.normalized);
                    float angleCriterion1 = Vector2.SignedAngle(-aEdgePre.getVector.normalized, -bEdgePre.getVector.normalized);
                    float angleCriterion2 = Vector2.SignedAngle(-aEdgePre.getVector.normalized, bEdgeNxt.getVector.normalized);
                    
                    if (angleCriterion0 < 0) 
                        angleCriterion0 += 360;
                    if  (angleCriterion1 < 0)
                        angleCriterion1 += 360;
                    if  (angleCriterion2 < 0)
                        angleCriterion2 += 360;
                    
                    // ab互穿，那么在两侧，angleCriterion0 不应是最大或最小
                    if (  ! ((angleCriterion0 > angleCriterion1 && angleCriterion0 > angleCriterion2) ||  
                             (angleCriterion0 < angleCriterion1 && angleCriterion0 < angleCriterion2))   )
                    {
                        // 然后判断，是入点还是出点
                        if (angleCriterion1 < angleCriterion2)
                        {
                            // 不管是求交还是求并，都是入点
                            inPoints.Add(newInoutPoint);
                        }
                        else
                        {
                            // 不管是求交还是求并，都是出点
                            outPoints.Add(newInoutPoint);
                        }
                    }
                    // 弹射而过，那么乘积为正
                    else
                    {
                        // 不管是求交还是求并，都不是出点或入点
                        // do nothing
                    }
                }
                
                // 情况二，同位交点有3个，说明是Y字型，其中一条是并线所以少一个
                else if (samePointInt.Count == 3)
                {
                    // 如果是分叉，即两个pre边的方向一致
                    if ((aEdgePre.getVector.normalized - bEdgePre.getVector.normalized).magnitude <=
                        GeometryConstant.Tolerance)
                    {
                        // 那么考察两个nxt边的情况
                        
                        // 考虑两组向量之间的逆时针夹角，-aEdgePre x aEdgeNxt 与 -aEdgePre x bEdgeNxt
                        // 若前者大，那么为可能的入点，若后者大，那么为可能的出点
                        float angleCriterion0 = Vector2.SignedAngle(-aEdgePre.getVector.normalized, bEdgeNxt.getVector.normalized);
                        float angleCriterion1 = Vector2.SignedAngle(-aEdgePre.getVector.normalized, aEdgeNxt.getVector.normalized);
                        if (angleCriterion0 < 0) 
                            angleCriterion0 += 360;
                        if  (angleCriterion1 < 0)
                            angleCriterion1 += 360;
                        
                        if (angleCriterion0 > angleCriterion1)
                        {
                            // 先是并线，然后a往外，b往内
                            if (IOA)
                            {
                                // 对于求交，这种情况算真入点
                                inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算伪入点
                                // inPoints.Add(newInoutPoint);
                            }
                        }
                        else
                        {
                            // 先是并线，然后a往内，b往外
                            if (IOA)
                            {
                                // 对于求交，这种情况算伪出点
                                // outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算真出点
                                outPoints.Add(newInoutPoint);
                            }
                        }
                    }
                    // 如果是合并，俩Nxt边方向一致
                    else if ((aEdgeNxt.getVector.normalized - bEdgeNxt.getVector.normalized).magnitude <=
                             GeometryConstant.Tolerance)
                    {
                        // 那么考察两个pre边的情况
                        
                        // 考虑两组向量之间的逆时针夹角，aEdgeNxt x -bEdgePre 与 aEdgeNxt x -aEdgePre
                        // 若前者大，那么为可能的入点，若后者大，那么为可能的出点
                        float angleCriterion0 = Vector2.SignedAngle(aEdgeNxt.getVector.normalized, -bEdgePre.getVector.normalized);
                        float angleCriterion1 = Vector2.SignedAngle(aEdgeNxt.getVector.normalized, -aEdgePre.getVector.normalized);
                        if (angleCriterion0 < 0) 
                            angleCriterion0 += 360;
                        if  (angleCriterion1 < 0)
                            angleCriterion1 += 360;
                        
                        if (angleCriterion0 > angleCriterion1)
                        {
                            // a从内，b从外，两者并线
                            if (IOA)
                            {
                                // 对于求交，这种情况算伪入点
                                // inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算真入点
                                inPoints.Add(newInoutPoint);
                            }
                        }
                        else
                        {
                            // a从外，b从内，两者并线
                            if (IOA)
                            {
                                // 对于求交，这种情况算真出点
                                outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算伪出点
                                // inPoints.Add(newInoutPoint);
                            }
                        }
                    }
                    // 情况三，aNxt和bPre共线反向
                    else if ((aEdgeNxt.getVector.normalized + bEdgePre.getVector.normalized).magnitude <=
                             GeometryConstant.Tolerance)
                    {
                        // 那么考察是入点还是出点：
                        // 考虑两组向量之间的逆时针夹角，aEdgeNxt x bEdgeNxt 与 aEdgeNxt x -aEdgePre
                        // 若前者大，那么为可能的出点，若后者大，那么为可能的入点
                        float angleCriterion0 = Vector2.SignedAngle(aEdgeNxt.getVector.normalized, bEdgeNxt.getVector.normalized);
                        float angleCriterion1 = Vector2.SignedAngle(aEdgeNxt.getVector.normalized, -aEdgePre.getVector.normalized);
                        if (angleCriterion0 < 0) 
                            angleCriterion0 += 360;
                        if  (angleCriterion1 < 0)
                            angleCriterion1 += 360;

                        if (angleCriterion0 > angleCriterion1)
                        {
                            // 从并线到b跑出去，可能的出点
                            if (IOA)
                            {
                                // 对于求交，是伪出点
                                // outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算真出点
                                outPoints.Add(newInoutPoint);
                            }
                        }
                        else
                        {
                            // 从并线到b跑进去，可能的入点
                            if (IOA)
                            {
                                // 对于求交，是真入点
                                inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算伪入点
                                // inPoints.Add(newInoutPoint);
                            }
                        }
                    }
                    // 情况四，aPre和bNxt共线反向
                    else if ((aEdgePre.getVector.normalized + bEdgeNxt.getVector.normalized).magnitude <=
                             GeometryConstant.Tolerance)
                    {
                        // 那么考察是入点还是出点：
                        // 考虑两组向量之间的逆时针夹角，bEdgeNxt x aEdgeNxt 与 bEdgeNxt x -bEdgePre
                        // 若前者大，那么为可能的入点，若后者大，那么为可能的出点
                        float angleCriterion0 = Vector2.SignedAngle(bEdgeNxt.getVector.normalized, aEdgeNxt.getVector.normalized);
                        float angleCriterion1 = Vector2.SignedAngle(bEdgeNxt.getVector.normalized, -bEdgePre.getVector.normalized);
                        if (angleCriterion0 < 0) 
                            angleCriterion0 += 360;
                        if  (angleCriterion1 < 0)
                            angleCriterion1 += 360;
                        
                        if (angleCriterion0 > angleCriterion1)
                        {
                            // 可能的入点
                            if (IOA)
                            {
                                // 对于求交，这种情况算伪入点
                                // inPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算真入点
                                inPoints.Add(newInoutPoint);
                            }
                        }
                        else
                        {
                            // 可能的出点
                            if (IOA)
                            {
                                // 对于求交，这种情况算真出点
                                outPoints.Add(newInoutPoint);
                            }
                            else
                            {
                                // 对于求并，这种情况算伪出点
                                // outPoints.Add(newInoutPoint);
                            }
                        }
                    }
                    
                }

                continue;
            }
        }
    }

    // 从所有的出入点，计算新Loops
    static public void CalculateLoopsFromInoutPoints(ref Mesh meshA, ref Mesh meshB, ref List<InOutPoint> inPoints, ref List<InOutPoint> outPoints, ref List<Loop> loops, bool IOA)
    {
        // 结果中的拓扑和原来的拓扑的对应关系
        Dictionary<Edge, Edge> ans2refEdges = new Dictionary<Edge, Edge>();
        Dictionary<Edge, Edge> ref2ansEdges = new Dictionary<Edge, Edge>();
        Dictionary<Vertex, Vertex> ans2refVertices = new Dictionary<Vertex, Vertex>();
        Dictionary<Vertex, Vertex> ref2ansVertices = new Dictionary<Vertex, Vertex>();
        
        // 所以为了搜索完备，给每个出点、入点打上访问标
        bool[] inPvisit = new bool[inPoints.Count];
        bool[] outPvisit = new bool[outPoints.Count];
        for (int i = 0; i < inPvisit.Length; i++)
            inPvisit[i] = false;
        for (int i = 0; i < outPvisit.Length; i++)
            outPvisit[i] = false;

        // 基于入点，开始不断构建
        for (int i = 0; i < inPoints.Count; i++)
        {
            if (inPvisit[i])
                continue;

            // 当前出发的入点
            var inP = inPoints[i];

            // 创建新Loop
            Loop newLoop = new Loop();

            // 在出入点处理阶段，已经吧所有的出入点“退化”成X型了
            if (inP.EdgeA[1] is null && inP.EdgeB[1] is null) // 该入点是X型交叉
            {
                // 起始点
                //Vertex currBeginVert = new Vertex(inP.Point);
                //Vertex currEndVert = new Vertex( IOA ? inP.EdgeB[0].VertexEnd.Point : inP.EdgeA[0].VertexEnd.Point);
                
                Vertex currBeginVert = new Vertex();
                Vertex currEndVert = new Vertex(inP.Point);

                //ans2refVertices.Add(currEndVert, IOA ? inP.EdgeB[0].VertexEnd : inP.EdgeA[0].VertexEnd);

                // 当前的ref边，求交一开始是B，求并则是A
                Edge refEdge = null;// = IOA ? inP.EdgeB[0] : inP.EdgeA[0];

                //newLoop.AddVertex(currBeginVert);

                // 每次遇到新的边都试图找是否存在交点，可能是入点集或出点集，具体哪个集存在这里
                // 由于我们从入点开始，所以初始当然是出点（不可能连续两个入点的）
                //List<InOutPoint> targetInoutPoints = outPoints;
                //bool[] targetInoutPVisit = outPvisit;
                List<InOutPoint> targetInoutPoints = inPoints;
                bool[] targetInoutPVisit = inPvisit;
                
                // 当前正在沿途的mesh，是a还是b？求交一开始是B，求并则是A
                Mesh currMesh = IOA ? meshB : meshA;
                
                // 上一个循环，所涉及到的出入点，一开始是null，遇到之后，就改为某个点，处理完（下个循环开始）之后，重置为null
                //InOutPoint lastInoutPoint = null;
                InOutPoint lastInoutPoint = inP;

                // 开始沿着一圈找：
                while (true)
                {
                    // 每次新加入一个点
                    newLoop.AddVertex(currEndVert);

                    // 开始找下一个边的两个端点
                    currBeginVert = currEndVert;

                    // 一般情况，下个边就是EdgeNext
                    if (lastInoutPoint is null)
                    {
                        // 拿到下一个边
                        refEdge = ans2refVertices[currBeginVert].EdgeNext;
                    }
                    // 特判一下，如果上一次循环是从出入点过来的，显然不存在EdgeNext，而且得特殊修改一下refEdge
                    else // if (lastInoutPoint is not null)
                    {
                        if (currMesh == meshA)
                        {
                            refEdge = lastInoutPoint.EdgeA[0];
                        }
                        else
                        {
                            refEdge = lastInoutPoint.EdgeB[0];
                        }

                        // 归于null
                        lastInoutPoint = null;

                        // 出/入 翻转
                        if (targetInoutPoints == inPoints)
                        {
                            targetInoutPoints = outPoints;
                            targetInoutPVisit = outPvisit;
                        }
                        else if (targetInoutPoints == outPoints)
                        {
                            targetInoutPoints = inPoints;
                            targetInoutPVisit = inPvisit;
                        }
                    }

                    // 判断一手：有没有出/入点？ 一条边可能有多个出/入点
                    List<int> possibleInoutPointIdx = new List<int>();

                    for (int j = 0; j < targetInoutPoints.Count; j++)
                    {
                        if (targetInoutPVisit[j])
                            continue;
                        // TODO 这里是否要多考虑一手：当出入点是 T字形 或 十字形 ？

                        // 有一个出入点记录的边数据正是当前的边，说明该出入点可能是下一个点
                        // 当前正在A网格上跑，所以要从EdgeA上面找；或者当前正在B网格上跑，所以要从EdgeB上面找
                        if ((currMesh == meshA && targetInoutPoints[j].EdgeA[0] == refEdge) ||
                            (currMesh == meshB && targetInoutPoints[j].EdgeB[0] == refEdge))
                        {
                            // 同时要求：不能再背后，而是在跟前，向前看，排除又更近的点在后面的情况导致错误
                            if (Vector2.Dot(targetInoutPoints[j].Point - currEndVert.Point, refEdge.getVector) >= 0)
                            {
                                possibleInoutPointIdx.Add(j);
                            }
                        }
                    }

                    // 现在，拿到了下一条边上是否存在出/入点

                    // 如果没有，那么直接不管，直接抓着下一个edge往下迭代
                    if (possibleInoutPointIdx.Count == 0)
                    {
                        currEndVert = new Vertex(refEdge.VertexEnd.Point);

                        ans2refVertices.Add(currEndVert, refEdge.VertexEnd);
                    }
                    // 如果有，那就得找那个最近的
                    else
                    {
                        int bestPointIdx = 0;
                        float minDistance = float.MaxValue;

                        // 找最近的
                        foreach (var index in possibleInoutPointIdx)
                        {
                            float dist = (targetInoutPoints[index].Point - currBeginVert.Point).magnitude;
                            if (dist < minDistance)
                            {
                                minDistance = dist;
                                bestPointIdx = index;
                            }
                        }

                        // 现在获取了最近的出/入点
                        var bestInoutPoint = targetInoutPoints[bestPointIdx];
                        // 特判：如果等于最初进入的点，那么这个loop搞定了
                        if (bestInoutPoint == inP)
                        {
                            break;
                        }

                        // 不是初始入点，那就是新的点，继续迭代
                        // 当然就加进去了
                        // 且标记一手
                        targetInoutPVisit[bestPointIdx] = true;
                        // 它就是我们的新的end点
                        currEndVert = new Vertex(targetInoutPoints[bestPointIdx].Point);

                        // 翻转mesh
                        if (currMesh == meshB)
                            currMesh = meshA;
                        else if (currMesh == meshA)
                            currMesh = meshB;

                        // 设置所用的出入点
                        lastInoutPoint = targetInoutPoints[bestPointIdx];
                    }
                    // OK了？
                }
            }

            // Loop搞定了
            // 加到meshC
            newLoop.GenerateEdges();

            loops.Add(newLoop);
        }
    }

    
    // 判断loop在loop组的内？还是外
    static public bool LoopInOrOutLoops(in List<Loop> backLoops, in Loop targLoop, out bool isIn)
    {
        
        bool flag = true;
        int ans = 0;
        
        // 给一个用于获取绕数的点集合
        // 所有的顶点+所有的中点
        List<Vector2> points = new List<Vector2>();
        foreach (var vert in targLoop.Vertices)
        {
            if (points.Count() == 0)
            {
                points.Add(vert.Point);
            }
            else
            {
                Vector2 p = points[points.Count() - 1];
                points.Add((p + vert.Point) / 2.0f);
                points.Add(vert.Point);
            }
        }
        
        // 只需要一个点获取正确的绕数就能得到
        foreach (var point in points)
        {
            // 360°，从0°开始，步长30°
            for (int i = 0; i < 12; i++)
            {
                flag = true;
                
                float thisAngleInRadian = (i / 12.0f) * (Mathf.PI * 2.0f);
                // 方向，根据三角函数计算
                Vector2 dir = new Vector2(Mathf.Cos(thisAngleInRadian), Mathf.Sin(thisAngleInRadian));
                
                // 这条射线
                Edge thisRay = new Edge();
                thisRay.VertexBegin = new Vertex(point);
                thisRay.VertexEnd = new Vertex(point + 150.0f * dir);
                
                List<IntersectionEdgeEdgeResult> results = new List<IntersectionEdgeEdgeResult>();
                // 开始计算，对背景loop每个边都
                foreach (var backLoop in backLoops)
                {
                    foreach (var edge in backLoop.Edges)
                    {
                        IntersectionEdgeEdgeResult currInt = null;
                        IntersectionEdgeEdge(edge, thisRay, out currInt);

                        // 真有相交情况
                        if (currInt is not null)
                        {
                            // X型单点相交
                            if (currInt.iType == IntersectionEdgeEdgeResult.intersectionType.SinglePoint &&
                                currInt.spsr == IntersectionEdgeEdgeResult.singlePointSpecialRelationship.None)
                                results.Add(currInt);
                            // 否则是重合边或T型
                            else
                            {
                                flag = false;
                                break;
                            }
                        }
                    }
                    if (!flag)
                        break;
                }

                if (!flag)
                    continue;

                // 全部是X型的正确交点
                // 看他们是奇数还是偶数
                else
                {
                    ans = results.Count & 1;
                    break;
                }
            }

            if (flag)
                break;
        }
        // 奇数说明点在多边形内部，否则在外部
        isIn = ans == 1;
        if (flag)
        {
            // 正常情况，有答案
            return true;
        }
        else{
            // 如果没有答案，也就是每个点都拿不到一个干净的无向绕数判断
            // 说明backLoops至少有一个loop和targLoop重合（同向或反向）
            // 需要进一步判断
            return false;
        }
    }
    
    
    // 当没有任何入点与出点时，计算新的Loops
    static public void CalculateLoopsWhenNoInoutPoints(ref Mesh meshA, ref Mesh meshB, ref int[] aLoopIntInfo, ref int[] bLoopIntInfo, ref List<Loop> loops, bool IOA)
    {
        // 每个loop：在对方多边形内部？还是外部？
        bool[] aLoopsInOrOutB = new bool[meshA.Loops.Count];
        bool[] bLoopsInOrOutA = new bool[meshB.Loops.Count];
        
        bool[] aLoopsHasOverlapInB = new bool[meshA.Loops.Count];
        bool[] bLoopsHasOverlapInA = new bool[meshB.Loops.Count];

        for(int i = 0; i < meshA.Loops.Count; i++)
        {
            var loop = meshA.Loops[i];
            // 是否存在重合边？该loop在对方多边形的内部还是外部？
            aLoopsHasOverlapInB[i] = LoopInOrOutLoops(in meshB.Loops, in loop, out aLoopsInOrOutB[i]);
        }
        for(int i = 0; i < meshB.Loops.Count; i++)
        {
            var loop = meshB.Loops[i];
            // 是否存在重合边？该loop在对方多边形的内部还是外部？
            bLoopsHasOverlapInA[i] = LoopInOrOutLoops(in meshA.Loops, in loop, out bLoopsInOrOutA[i]);
        }
        
        loops.Clear();
        // 根据包围或被包围的情况，向结果loop添加
        for (int i = 0; i < meshA.Loops.Count; i++)
        {
            // 该Loop存在交点或者重合了，那别管
            if (aLoopIntInfo[i] != 0)
            {
                continue;
            }
            Loop currLoop = meshA.Loops[i];
            if (aLoopsInOrOutB[i])
            {
                if (IOA)
                {
                    // 求交时，若A环包在B内，则保留
                    loops.Add(currLoop);
                }
                // 求并时，若A环包在B内，则放弃
            }
            else
            {
                // 求并时，若A环在B外，则保留
                if (!IOA)
                {
                    loops.Add(currLoop);
                }
                // 求交时，若A环在B外，则放弃
            }
        }
        for (int i = 0; i < meshB.Loops.Count; i++)
        {
            // 该Loop存在交点或重合了
            if (bLoopIntInfo[i] != 0)
            {
                continue;
            }
            Loop currLoop = meshB.Loops[i];
            if (bLoopsInOrOutA[i])
            {
                if (IOA)
                {
                    // 求交时，若B环包在A内，则保留
                    loops.Add(currLoop);
                }
                // 求并时，若B环包在A内，则放弃
            }
            else
            {
                // 求并时，若B环在A外，则保留
                if (!IOA)
                {
                    loops.Add(currLoop);
                }
                // 求交时，若B环在A外，则放弃
            }
        }
    }
    
    // 根据重合loop构建loop结果
    static public void CalculateLoopsFromSameLoops(ref List<LoopInSameInfo> loopInSameInfos, ref List<Loop> loops /*, bool IOA*/)
    {
        // 对于每一个重叠组
        foreach (var loopInSameInfo in loopInSameInfos)
        {
            // 判断LoopA、LoopB分别是顺时针还是逆时针
            float sumLoopACross = 0, sumLoopBCross = 0;
            
            for (int i = 0; i < loopInSameInfo.loopA.Vertices.Count; i++)
            {
                int preIdx = i, 
                    thiIdx = (i + 1) % loopInSameInfo.loopA.Vertices.Count,
                    nxtIdx = (i + 2) % loopInSameInfo.loopA.Vertices.Count;
                
                Vector2 thiVec = loopInSameInfo.loopA.Vertices[thiIdx].Point - 
                                 loopInSameInfo.loopA.Vertices[preIdx].Point;
                Vector2 nxtVec = loopInSameInfo.loopA.Vertices[nxtIdx].Point - 
                                 loopInSameInfo.loopA.Vertices[thiIdx].Point;

                sumLoopACross += Cross(thiVec, nxtVec);
            }
            for (int i = 0; i < loopInSameInfo.loopB.Vertices.Count; i++)
            {
                int preIdx = i, 
                    thiIdx = (i + 1) % loopInSameInfo.loopB.Vertices.Count,
                    nxtIdx = (i + 2) % loopInSameInfo.loopB.Vertices.Count;
                
                Vector2 thiVec = loopInSameInfo.loopB.Vertices[thiIdx].Point - 
                                 loopInSameInfo.loopB.Vertices[preIdx].Point;
                Vector2 nxtVec = loopInSameInfo.loopB.Vertices[nxtIdx].Point - 
                                 loopInSameInfo.loopB.Vertices[thiIdx].Point;

                sumLoopBCross += Cross(thiVec, nxtVec);
            }
            
            // sumLoopACross为正，表示LoopA是正向（逆时针）环，否则为负向（顺时针）环
            // sumLoopBCross为正，表示LoopB是正向（逆时针）环，否则为负向（顺时针）环
            
            // 法则：
            // 1 如果AB为同向环，那么无论交并，都保留改环
            // 2 如果AB为反向环，那么无论交并，都完全去除
            if (sumLoopACross * sumLoopBCross >= 0)
                loops.Add(loopInSameInfo.loopA);
            else
            {
                // DO NOTHING
            }
        }
    }
    
    // 将A∩B结果存储于C
    static public void Intersection(ref Mesh meshA, ref Mesh meshB, ref Mesh meshC)
    {
        // 拿到所有的交点信息，在本算法设计中，重合边可以放弃
        List<IntersectionEdgeEdgeResult> intersectionResults = new List<IntersectionEdgeEdgeResult>();

        // 求出所有的相交情况
        getAllEdgeIntersectionOfTwoMesh(ref meshA, ref meshB, ref intersectionResults);

        // 入点或广义入点
        List<InOutPoint> inPoints = new List<InOutPoint>();
        // 出点或广义出点
        List<InOutPoint> outPoints = new List<InOutPoint>();

        CalculateInoutPoint(ref intersectionResults, ref inPoints, ref outPoints, true);

        // Debug:打印所有入、出点
        /*Debug.Log("入点：");
        foreach (var p in inPoints)
        {
            Debug.Log(p.Point);
        }

        Debug.Log("出点：");
        foreach (var p in outPoints)
        {
            Debug.Log(p.Point);
        }*/

        // 开始构建交集，最终，每个出点和入点都会被经过
        // 结果的Loop集合
        List<Loop> loops0 = new List<Loop>();   // 重合环交并结果
        List<Loop> loops1 = new List<Loop>();   // 无交点环交并结果
        List<Loop> loops2 = new List<Loop>();   // 有交点环交并结果
        
        // 0：该idx的loop的边没有任何交点
        // 1：该idx的loop的边有交点
        // -1：该idx的loop和对方某个loop完全重合了
        int[] aloopIntInfo = new int[meshA.Loops.Count];
        int[] bloopIntInfo = new int[meshB.Loops.Count];
        for (int i = 0; i < aloopIntInfo.Length; i++)
        {
            aloopIntInfo[i] = 0;
        }
        for (int i = 0; i < bloopIntInfo.Length; i++)
        {
            bloopIntInfo[i] = 0;
        }
        // 通过已有的出入点信息，计算有哪些loop包含了出入点
        for (int i = 0; i < inPoints.Count; i++)
        {
            // 每个入点所在的loop都标记一下
            // 找出那个loop
            for (int j = 0; j < meshA.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshA.Loops[j] == inPoints[i].EdgeA[0].PLoop)
                {
                    aloopIntInfo[j] = 1;
                }
            }
            for (int j = 0; j < meshB.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshB.Loops[j] == inPoints[i].EdgeB[0].PLoop)
                {
                    bloopIntInfo[j] = 1;
                }
            }
        }
        for (int i = 0; i < outPoints.Count; i++)
        {
            // 每个入点所在的loop都标记一下
            // 找出那个loop
            for (int j = 0; j < meshA.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshA.Loops[j] == outPoints[i].EdgeA[0].PLoop)
                {
                    aloopIntInfo[j] = 1;
                }
            }
            for (int j = 0; j < meshB.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshB.Loops[j] == outPoints[i].EdgeB[0].PLoop)
                {
                    bloopIntInfo[j] = 1;
                }
            }
        }
        
        
        // 还有一件事情：找出A和B里面的重合边
        List<LoopInSameInfo> loopInSameInfos =  new List<LoopInSameInfo>();
        for (int i = 0; i < meshA.Loops.Count; i++)
        {
            for (int j = 0; j < meshB.Loops.Count; j++)
            {
                if (TwoLoopInSame(meshA.Loops[i], meshB.Loops[j]))
                {
                    aloopIntInfo[i] = -1;
                    bloopIntInfo[j] = -1;
                    
                    LoopInSameInfo newloopInSame = new LoopInSameInfo();
                    newloopInSame.loopA = meshA.Loops[i];
                    newloopInSame.loopB = meshB.Loops[j];
                    
                    loopInSameInfos.Add(newloopInSame);
                }
            }
        }
        // 先计算重合边们的交并情况
        CalculateLoopsFromSameLoops(ref loopInSameInfos, ref loops0);
        
        // 如果不存在出点和入点都没有，那么采用内外法构建
        CalculateLoopsWhenNoInoutPoints(ref meshA, ref meshB, ref aloopIntInfo, ref bloopIntInfo, ref loops1, true);
        
        if (inPoints.Count != 0 && outPoints.Count != 0)
        {
            // 从出入点集合，计算loops
            CalculateLoopsFromInoutPoints(ref meshA, ref meshB, ref inPoints, ref outPoints, ref loops2, true);
        }

        foreach (var loop in Enumerable.Concat(loops0, Enumerable.Concat(loops1, loops2)))
        {
            meshC.AddLoop(loop);
            // meshC.OptimizeTopo();
            meshC.needUpdate = true;
        }
        
        
        /*
         * // 单点、段重合两种情况不同处理

         */
    }
    
    
    // 将A∪B结果存储于C
    static public void Addition(ref Mesh meshA, ref Mesh meshB, ref Mesh meshC)
    {
        // 拿到所有的交点信息，在本算法设计中，重合边可以放弃
        List<IntersectionEdgeEdgeResult> intersectionResults = new List<IntersectionEdgeEdgeResult>();

        // 求出所有的相交情况
        getAllEdgeIntersectionOfTwoMesh(ref meshA, ref meshB, ref intersectionResults);

        // 入点或广义入点
        List<InOutPoint> inPoints = new List<InOutPoint>();
        // 出点或广义出点
        List<InOutPoint> outPoints = new List<InOutPoint>();
        
        CalculateInoutPoint(ref intersectionResults, ref inPoints, ref outPoints, false);
        
        
        // 开始构建交集，最终，每个出点和入点都会被经过
        // 结果的Loop集合
        List<Loop> loops0 = new List<Loop>();   // 重合环交并结果
        List<Loop> loops1 = new List<Loop>();   // 无交点环交并结果
        List<Loop> loops2 = new List<Loop>();   // 有交点环交并结果
        
        // 0：该idx的loop的边没有任何交点
        // 1：该idx的loop的边有交点
        // -1：该idx的loop和对方某个loop完全重合了
        int[] aloopIntInfo = new int[meshA.Loops.Count];
        int[] bloopIntInfo = new int[meshB.Loops.Count];
        for (int i = 0; i < aloopIntInfo.Length; i++)
        {
            aloopIntInfo[i] = 0;
        }
        for (int i = 0; i < bloopIntInfo.Length; i++)
        {
            bloopIntInfo[i] = 0;
        }
        // 通过已有的出入点信息，计算有哪些loop包含了出入点
        for (int i = 0; i < inPoints.Count; i++)
        {
            // 每个入点所在的loop都标记一下
            // 找出那个loop
            for (int j = 0; j < meshA.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshA.Loops[j] == inPoints[i].EdgeA[0].PLoop)
                {
                    aloopIntInfo[j] = 1;
                }
            }
            for (int j = 0; j < meshB.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshB.Loops[j] == inPoints[i].EdgeB[0].PLoop)
                {
                    bloopIntInfo[j] = 1;
                }
            }
        }
        for (int i = 0; i < outPoints.Count; i++)
        {
            // 每个入点所在的loop都标记一下
            // 找出那个loop
            for (int j = 0; j < meshA.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshA.Loops[j] == outPoints[i].EdgeA[0].PLoop)
                {
                    aloopIntInfo[j] = 1;
                }
            }
            for (int j = 0; j < meshB.Loops.Count; j++)
            {
                // 如果这个loop和这个入点
                if (meshB.Loops[j] == outPoints[i].EdgeB[0].PLoop)
                {
                    bloopIntInfo[j] = 1;
                }
            }
        }
        
        
        // 还有一件事情：找出A和B里面的重合边
        List<LoopInSameInfo> loopInSameInfos =  new List<LoopInSameInfo>();
        for (int i = 0; i < meshA.Loops.Count; i++)
        {
            for (int j = 0; j < meshB.Loops.Count; j++)
            {
                if (TwoLoopInSame(meshA.Loops[i], meshB.Loops[j]))
                {
                    aloopIntInfo[i] = -1;
                    bloopIntInfo[j] = -1;
                    
                    LoopInSameInfo newloopInSame = new LoopInSameInfo();
                    newloopInSame.loopA = meshA.Loops[i];
                    newloopInSame.loopB = meshB.Loops[j];
                    
                    loopInSameInfos.Add(newloopInSame);
                }
            }
        }
        // 先计算重合边们的交并情况
        CalculateLoopsFromSameLoops(ref loopInSameInfos, ref loops0);
        
        
        // 如果不存在出点和入点都没有，那么采用内外法构建
        CalculateLoopsWhenNoInoutPoints(ref meshA, ref meshB, ref aloopIntInfo, ref bloopIntInfo, ref loops1, false);
        
        if (inPoints.Count != 0 && outPoints.Count != 0)
        {
            // 从出入点集合，计算loops
            CalculateLoopsFromInoutPoints(ref meshA, ref meshB, ref inPoints, ref outPoints, ref loops2, false);
        }

        foreach (var loop in Enumerable.Concat(loops0, Enumerable.Concat(loops1, loops2)))
        {
            meshC.AddLoop(loop);
            meshC.OptimizeTopo();
            meshC.needUpdate = true;
        }
    }
}