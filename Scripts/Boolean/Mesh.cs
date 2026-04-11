using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GeometryConstant
{
    static public float Tolerance =  1e-5f;
}

public class Vertex
{
    public Loop PLoop;
    public Vector2 Point;
    public Edge EdgeLast;
    public Edge EdgeNext;

    public Vertex()
    {
        Point = new Vector2(0, 0);
        EdgeLast = null;
        EdgeNext = null;
    }
    public Vertex(float x, float y)
    {
        Point = new Vector2(x, y);
    }
    public Vertex(Vector2 point)
    {
        Point = point;
    }

    public void SetPoint(Vector2 point)
    {
        Point = point;
        //Update();
    }

    public void SetPoint(float x, float y)
    {
        Point = new Vector2(x, y);
        //Update();
    }

    /*private void Update()
    {
        PLoop.Update();
    }*/
}

public class Edge
{
    public Vertex VertexBegin;
    public Vertex VertexEnd;

    public Vector2 getVector
    {
        get
        {
            return VertexEnd.Point - VertexBegin.Point;
        }
    }

    public Loop PLoop;
}

public class Loop
{
    public Mesh PMesh;
    public List<Vertex> Vertices = new List<Vertex>();
    public List<Edge> Edges = new List<Edge>();

    public void AddVertex(Vertex vertex)
    {
        vertex.PLoop = this;
        // 判断是否和第一个点重合
        foreach (Vertex v in Vertices)
        {
            if (Vector2.Distance(vertex.Point, v.Point) <= GeometryConstant.Tolerance)
                return;
        }
        // 不重合，方才添加进去
        Vertices.Add(vertex);
    }

    public void AddEdge(Vertex a, Vertex b)
    {
        Edge newEdge = new Edge();
        newEdge.VertexBegin = a;
        newEdge.VertexEnd = b;
        newEdge.PLoop = this;
        a.EdgeNext = newEdge;
        b.EdgeLast = newEdge;
        
        Edges.Add(newEdge);
    }

    /*public void Update()
    {
        PMesh.Update();
    }*/

    public Vector3[] GetPositions()
    {
        Vector3[] positions = new Vector3[Vertices.Count + 1];
        for (int i = 0; i < positions.Length; i++)
        {
            positions[i] = Vertices[i % Vertices.Count].Point;
        }
        
        return positions;
    }

    public void GenerateEdges()
    {
        Edges.Clear();
        for (int j = 0; j < Vertices.Count; j++)
        {
            AddEdge(Vertices[j % Vertices.Count], Vertices[(j + 1) % Vertices.Count]);
        }
    }
}

public class Mesh:MonoBehaviour
{
    public bool needUpdate = false;
    
    public List<LineRenderer> LineRenderers = new List<LineRenderer>();
    
    public List<Loop> Loops = new List<Loop>();
    public void AddLoop(Loop loop)
    {
        loop.PMesh = this;
        Loops.Add(loop);
        //Update();
    }
    
    // 优化拓扑
    public void OptimizeTopo()
    {
        // 删除所有小于两个顶点的loop
        List<Loop> deleteLoops = new List<Loop>();
        foreach (Loop loop in Loops)
        {
            if (loop.Vertices.Count <= 2)
            {
                deleteLoops.Add(loop);
            }
        }
        foreach (Loop loop in deleteLoops)
        {
            Loops.Remove(loop);
        }
        
        // 有些Loop是否重合边，需要处理、拆分，将变化后的edge弄好并记录loop
        List<Loop> splitLoops = new List<Loop>();
        foreach (Loop loop in Loops)
        {
            // loop内部可能的相互重叠情形
            List<BooleanOperation.IntersectionEdgeEdgeResult> results = new List<BooleanOperation.IntersectionEdgeEdgeResult>();
            // 边之间，两两测试
            for (int i = 0; i < loop.Edges.Count; i++)
            {
                for (int j = i + 1; j < loop.Edges.Count; j++)
                {
                    // 尝试求交
                    BooleanOperation.IntersectionEdgeEdgeResult newResult = null;
                    BooleanOperation.IntersectionEdgeEdge(loop.Edges[i], loop.Edges[j], out newResult);
                    
                    // 若有则保存下来
                    if (newResult is not null && newResult.iType == BooleanOperation.IntersectionEdgeEdgeResult.intersectionType.SegmentOverlap)
                    {
                        results.Add(newResult);
                    }
                }
            }

            // 很正常的边
            if (results.Count == 0)
            {
                continue;
            }
            else
            {
                splitLoops.Add(loop);
            }
            
            // 现在拿到了loop自重叠
            // 涉及自重叠的边，都要删去自重叠的部分
            List<Edge> deleteEdges = new List<Edge>();
            List<Edge> newEdges = new List<Edge>();
            foreach (var result in results)
            {
                // 和A边的方向如何
                float dirAnsA = Vector2.Dot(result.aEdge.getVector, result.intEdge.getVector);
                // 交边和A边方向一致
                if (dirAnsA > 0) 
                {
                    // 新增边
                    Edge newEdgeBegin =  new Edge();
                    newEdgeBegin.VertexBegin = result.aEdge.VertexBegin;
                    // 该点是新增的
                    newEdgeBegin.VertexEnd = result.intEdge.VertexBegin;
                    
                    Edge newEdgeEnd =  new Edge();
                    // 该点是新增的
                    newEdgeEnd.VertexBegin = result.intEdge.VertexEnd;
                    newEdgeEnd.VertexEnd = result.aEdge.VertexEnd;
                    
                    // 如果不是退化边，就加上去
                    if (newEdgeBegin.getVector.magnitude > GeometryConstant.Tolerance)
                    {
                        newEdges.Add(newEdgeBegin);
                    }
                    if (newEdgeEnd.getVector.magnitude > GeometryConstant.Tolerance)
                    {
                        newEdges.Add(newEdgeEnd);
                    }
                }
                
                // 和B边的方向如何
                float dirAnsB = Vector2.Dot(result.bEdge.getVector, result.intEdge.getVector);
                if (dirAnsB > 0) 
                {
                    // 新增边
                    Edge newEdgeBegin =  new Edge();
                    newEdgeBegin.VertexBegin = result.bEdge.VertexBegin;
                    // 该点是新增的
                    newEdgeBegin.VertexEnd = result.intEdge.VertexBegin;
                    
                    Edge newEdgeEnd =  new Edge();
                    // 该点是新增的
                    newEdgeEnd.VertexBegin = result.intEdge.VertexEnd;
                    newEdgeEnd.VertexEnd = result.bEdge.VertexEnd;
                    
                    // 如果不是退化边，就加上去
                    if (newEdgeBegin.getVector.magnitude > GeometryConstant.Tolerance)
                    {
                        newEdges.Add(newEdgeBegin);
                    }
                    if (newEdgeEnd.getVector.magnitude > GeometryConstant.Tolerance)
                    {
                        newEdges.Add(newEdgeEnd);
                    }
                }
                
                // A、B边都要去除
                deleteEdges.Add(result.aEdge);
                deleteEdges.Add(result.bEdge);
            }
            // 删掉分裂的边
            foreach (Edge edge in deleteEdges)
            {
                loop.Edges.Remove(edge);
            }
            // 将新增边加进去
            loop.Edges.AddRange(newEdges);
            
            // 记录这个将受改变的loop
            splitLoops.Add(loop);
        }
        
        // 此时，edge已经对了，但是vertex需要改、顺序需要改、loop也有可能一分为二
        // TODO 重整Loops，不过不整的话，渲染效果也不会有问题
        // loop里面已经加好了新的边了
        List<Loop> newLoops = new List<Loop>();
        foreach (Loop loop in splitLoops)
        {
            // 开始对edges建图
            bool[] edgeVisit = new bool[loop.Edges.Count];
            for (int i = 0; i < edgeVisit.Length; i++)
            {
                edgeVisit[i] = false;
            }

            // 遍历建图
            for (int i = 0; i < loop.Edges.Count; i++)
            {
                if (edgeVisit[i])
                    continue;
                
                // 当前的图（环）
                List<Edge> newEdgeList = new List<Edge>();
                // 起始的edge
                Edge currEdge = loop.Edges[i];
                while (true)
                {
                    newEdgeList.Add(currEdge);
                    
                    bool thereIsANextEdge = false;
                    // 找到相连接的edge
                    for (int j = 0; j < loop.Edges.Count; j++)
                    {
                        if (edgeVisit[j])
                            continue;

                        // 找到那个以头接我尾的那条边
                        if ((loop.Edges[j].VertexBegin.Point - currEdge.VertexEnd.Point).magnitude <
                            GeometryConstant.Tolerance)
                        {
                            // 若数据合法，必然有一个，否则就是环结束了。
                            currEdge = loop.Edges[j];
                            edgeVisit[j] = true;
                            thereIsANextEdge = true;
                            break;
                        }
                    }
                    // 找到了下一条边
                    if (thereIsANextEdge)
                    {
                        continue;
                    }
                    // 如果没有找到下一条边，就认为环搞完了
                    else
                    {
                        break;
                    }
                }
                
                // 拿到了这个环的newEdgeList
                Loop newLoop = new Loop();
                // 加入第一个点
                Vertex newBeginVertex = new Vertex();
                newBeginVertex.Point = newEdgeList[0].VertexBegin.Point;
                newLoop.AddVertex(newBeginVertex);
                
                // 加入后面的每个点
                for (int j = 0; j < newEdgeList.Count; j++)
                {
                    Vertex newVertex = new Vertex();
                    newVertex.Point = newEdgeList[j].VertexEnd.Point;
                    newLoop.AddVertex(newVertex);
                }
                
                // 得到了所有的点，生成边
                newLoop.GenerateEdges();
                // 加入之
                newLoops.Add(newLoop);
            }
        }
        
        // 删除splitLoops
        foreach (Loop loop in splitLoops)
        {
            Loops.Remove(loop);
        }
        // 加入所有的新loop
        Loops.AddRange(newLoops);

        needUpdate = true;
    }

    public void ClearAll()
    {
        foreach (LineRenderer lr in LineRenderers)
        {
            // 删去所有的lineRenderer，并删去挂接它的物体
            GameObject go = lr.gameObject;
            Destroy(lr);
            Destroy(go);
        }
        LineRenderers.Clear();

        Loops.Clear();
        
        needUpdate = true;
    }

    public void EnableDraw()
    {
        foreach (LineRenderer lr in LineRenderers)
        {
            lr.enabled = true;
        }
    }

    public void DisableDraw()
    {
        foreach (LineRenderer lr in LineRenderers)
        {
            lr.enabled = false;
        }
    }


    public static void GetReverse(Mesh resourceMesh, ref Mesh newMesh)
    {
        newMesh.ClearAll();
        foreach (var loop in resourceMesh.Loops)
        {
            Loop newLoop = new Loop();
            int count = loop.Edges.Count;
            for (int i = 0; i < count; i++)
            {
                // 反向构建
                Vertex newVertex = new Vertex();
                newVertex.Point = loop.Vertices[count - 1 - i].Point;
                newLoop.AddVertex(newVertex);
            }
            newLoop.GenerateEdges();
            newMesh.AddLoop(newLoop);
        }
    }
}
