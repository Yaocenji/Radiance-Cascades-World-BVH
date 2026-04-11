#if UNITY_EDITOR

using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using RadianceCascadesWorldBVH;

// Boolean/Mesh.cs 在全局命名空间，与 UnityEngine.Mesh 冲突，用别名区分
using BoolMesh = global::Mesh;

namespace RadianceCascadesWorldBVH.Editor
{
    /// <summary>
    /// 编辑器工具：对两组 RCWBObject 执行批量布尔减法。
    ///
    /// 操作语义：A 组中每个物体的轮廓，依次减去 B 组中所有物体的轮廓，
    ///   结果写回该 A 物体的 ContourProfile。B 组物体不受影响。
    ///
    /// 单次减法原理：A_i - B_j = A_i ∩ (¬inflated_B_j)
    ///   先将 B_j 的 BoolMesh 沿顶点外法线膨胀 offsetEpsilon（Miter Join），
    ///   使减法结果与原始 B_j 之间保留等距细缝，再执行布尔交集。
    /// </summary>
    public class RCWBBooleanSubtractTool : EditorWindow
    {
        // ─── 物体组 ──────────────────────────────────────────────────────────
        private List<RCWBObject> m_GroupA = new List<RCWBObject>();
        private List<RCWBObject> m_GroupB = new List<RCWBObject>();

        // ─── 膨胀参数 ────────────────────────────────────────────────────────
        // offsetEpsilon：目标垂直间距（世界空间单位）。
        //   先将 B 沿顶点外法线膨胀此距离，使 A-B 结果与原始 B 之间保留等距细缝。
        // miterLimit：Miter 偏移倍数上限（= d_max / epsilon）。
        //   限制极尖顶角处的膨胀量，防止产生过大尖刺。
        private float m_OffsetEpsilon = 0.005f;
        private float m_MiterLimit    = 4f;

        // ─── UI 状态 ─────────────────────────────────────────────────────────
        private Vector2 m_ScrollA;
        private Vector2 m_ScrollB;

        [MenuItem("Tools/RCWB/Boolean Subtract (A - B)")]
        public static void ShowWindow()
        {
            GetWindow<RCWBBooleanSubtractTool>("Boolean Subtract");
        }

        // ─── 生命周期 ────────────────────────────────────────────────────────

        private void OnEnable()
        {
            if (m_GroupA == null) m_GroupA = new List<RCWBObject>();
            if (m_GroupB == null) m_GroupB = new List<RCWBObject>();
        }

        // ─── UI ──────────────────────────────────────────────────────────────

        private void OnGUI()
        {
            GUILayout.Label("RCWB 布尔减法工具", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "A 组每个物体的轮廓，依次减去 B 组所有物体的轮廓，结果写回 A 物体的 ContourProfile。\n" +
                "B 组物体不受影响。",
                MessageType.None);

            EditorGUILayout.Space();

            // ── Group A ──
            EditorGUILayout.LabelField("Group A（被减，结果写回）", EditorStyles.boldLabel);
            m_ScrollA = EditorGUILayout.BeginScrollView(m_ScrollA, GUILayout.MaxHeight(120));
            DrawObjectList(m_GroupA);
            EditorGUILayout.EndScrollView();
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("+ 添加空槽"))
                    m_GroupA.Add(null);
                if (GUILayout.Button("从选择填入 A"))
                    FillFromSelection(m_GroupA);
            }

            EditorGUILayout.Space();

            // ── Group B ──
            EditorGUILayout.LabelField("Group B（减去）", EditorStyles.boldLabel);
            m_ScrollB = EditorGUILayout.BeginScrollView(m_ScrollB, GUILayout.MaxHeight(120));
            DrawObjectList(m_GroupB);
            EditorGUILayout.EndScrollView();
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("+ 添加空槽"))
                    m_GroupB.Add(null);
                if (GUILayout.Button("从选择填入 B"))
                    FillFromSelection(m_GroupB);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            // ── 膨胀参数 ──
            EditorGUILayout.LabelField("膨胀设置（Inflate B）", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "执行前先将 B 的每个轮廓沿顶点外法线膨胀 Epsilon，\n" +
                "使减法结果与 B 之间保留等距细缝（Miter Join）。",
                MessageType.None);
            m_OffsetEpsilon = EditorGUILayout.FloatField(
                new GUIContent("Epsilon（世界单位）",
                    "膨胀目标垂直间距。典型值：0.001 ~ 0.01。"),
                m_OffsetEpsilon);
            m_OffsetEpsilon = Mathf.Max(0f, m_OffsetEpsilon);
            m_MiterLimit = EditorGUILayout.Slider(
                new GUIContent("Miter 上限",
                    "最大偏移倍数（d_max = Epsilon × Miter上限）。\n" +
                    "限制极尖顶角处的膨胀量，防止产生过大尖刺。"),
                m_MiterLimit, 1f, 20f);

            EditorGUILayout.Space();

            // ── 验证 & 执行 ──
            string error = Validate();
            if (error != null)
                EditorGUILayout.HelpBox(error, MessageType.Warning);

            EditorGUI.BeginDisabledGroup(error != null);
            int aCount = CountValid(m_GroupA);
            int bCount = CountValid(m_GroupB);
            if (GUILayout.Button(
                    $"执行：{aCount} 个 A 物体，各减去 {bCount} 个 B 物体",
                    GUILayout.Height(36)))
            {
                Execute(m_GroupA, m_GroupB, m_OffsetEpsilon, m_MiterLimit);
            }
            EditorGUI.EndDisabledGroup();
        }

        // ─── UI 辅助 ─────────────────────────────────────────────────────────

        private static void DrawObjectList(List<RCWBObject> list)
        {
            if (list.Count == 0)
            {
                EditorGUILayout.LabelField("（空，请添加物体）", EditorStyles.miniLabel);
                return;
            }
            for (int i = 0; i < list.Count; i++)
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    list[i] = (RCWBObject)EditorGUILayout.ObjectField(
                        list[i], typeof(RCWBObject), true);

                    string srcLabel = list[i] != null
                        ? GetContourSourceShort(list[i])
                        : "—";
                    EditorGUILayout.LabelField(srcLabel,
                        EditorStyles.miniLabel, GUILayout.Width(80));

                    if (GUILayout.Button("−", GUILayout.Width(22)))
                    {
                        list.RemoveAt(i);
                        GUIUtility.ExitGUI();
                    }
                }
            }
        }

        private static void FillFromSelection(List<RCWBObject> list)
        {
            foreach (var go in Selection.gameObjects)
            {
                var r = go.GetComponent<RCWBObject>();
                if (r != null && !list.Contains(r))
                    list.Add(r);
            }
        }

        private static string GetContourSourceShort(RCWBObject obj)
        {
            if (obj.ContourProfile != null && obj.ContourProfile.IsValid())
                return $"Profile({obj.ContourProfile.LoopCount})";
            var sr = obj.GetComponent<SpriteRenderer>();
            if (sr != null && sr.sprite != null && sr.sprite.GetPhysicsShapeCount() > 0)
                return "Sprite";
            return "无轮廓";
        }

        private static int CountValid(List<RCWBObject> list)
        {
            int n = 0;
            foreach (var o in list) if (o != null) n++;
            return n;
        }

        // ─── 验证 ────────────────────────────────────────────────────────────

        private string Validate()
        {
            if (CountValid(m_GroupA) == 0)
                return "Group A 中没有有效物体。";
            if (CountValid(m_GroupB) == 0)
                return "Group B 中没有有效物体。";

            foreach (var obj in m_GroupA)
            {
                if (obj == null) continue;
                if (!HasContourSource(obj))
                    return $"A 中的 '{obj.gameObject.name}' 没有有效轮廓来源。";
                if (m_GroupB.Contains(obj))
                    return $"'{obj.gameObject.name}' 同时在 A 和 B 中，请检查。";
            }
            foreach (var obj in m_GroupB)
            {
                if (obj == null) continue;
                if (!HasContourSource(obj))
                    return $"B 中的 '{obj.gameObject.name}' 没有有效轮廓来源。";
            }
            return null;
        }

        private static bool HasContourSource(RCWBObject obj)
        {
            if (obj.ContourProfile != null && obj.ContourProfile.IsValid())
                return true;
            var sr = obj.GetComponent<SpriteRenderer>();
            return sr != null && sr.sprite != null && sr.sprite.GetPhysicsShapeCount() > 0;
        }

        // ─── 核心执行 ────────────────────────────────────────────────────────

        private static void Execute(
            List<RCWBObject> groupA, List<RCWBObject> groupB,
            float offsetEpsilon, float miterLimit)
        {
            // 统计有效 A 物体，跳过 null
            var validA = new List<RCWBObject>();
            foreach (var o in groupA) if (o != null) validA.Add(o);
            var validB = new List<RCWBObject>();
            foreach (var o in groupB) if (o != null) validB.Add(o);

            int successCount = 0;
            int failCount    = 0;

            // 对每个 A 物体，逐一减去所有 B 物体
            for (int ai = 0; ai < validA.Count; ai++)
            {
                RCWBObject objA = validA[ai];

                EditorUtility.DisplayProgressBar(
                    "RCWB Boolean Subtract",
                    $"处理 {objA.gameObject.name} ({ai + 1}/{validA.Count})",
                    (float)ai / validA.Count);

                bool ok = ExecuteSingleA(objA, validB, offsetEpsilon, miterLimit);
                if (ok) successCount++;
                else    failCount++;
            }

            EditorUtility.ClearProgressBar();

            string summary = $"[RCWB Boolean] 完成：{successCount} 个 A 物体成功";
            if (failCount > 0) summary += $"，{failCount} 个失败（详见 Console）";
            Debug.Log(summary);

            if (failCount > 0)
                EditorUtility.DisplayDialog("部分失败",
                    $"成功 {successCount} 个，失败 {failCount} 个。\n详见 Console。", "确定");
        }

        /// <summary>
        /// 对单个 A 物体，顺序减去 validB 中每个 B 物体，写回 ContourProfile。
        /// 每次减法用上一次的结果作为新的 A 输入，实现链式相减。
        /// </summary>
        private static bool ExecuteSingleA(
            RCWBObject objA, List<RCWBObject> validB,
            float offsetEpsilon, float miterLimit)
        {
            // 用一个列表追踪所有临时 GO，确保 finally 中统一销毁
            var tempGOs = new List<GameObject>();

            try
            {
                // 初始 A 的世界空间 Mesh
                var goCurrentA = new GameObject("__BoolTemp_A") { hideFlags = HideFlags.HideAndDontSave };
                tempGOs.Add(goCurrentA);
                BoolMesh meshCurrentA = BuildWorldMesh(objA, goCurrentA);

                if (meshCurrentA.Loops.Count == 0)
                {
                    Debug.LogWarning($"[RCWB Boolean] 跳过 '{objA.gameObject.name}'：无法获取有效轮廓。");
                    return false;
                }

                // 链式减去每个 B
                foreach (var objB in validB)
                {
                    var goB    = new GameObject("__BoolTemp_B")     { hideFlags = HideFlags.HideAndDontSave };
                    var goBRev = new GameObject("__BoolTemp_B_Rev") { hideFlags = HideFlags.HideAndDontSave };
                    var goC    = new GameObject("__BoolTemp_C")     { hideFlags = HideFlags.HideAndDontSave };
                    tempGOs.Add(goB);
                    tempGOs.Add(goBRev);
                    tempGOs.Add(goC);

                    BoolMesh meshB    = BuildWorldMesh(objB, goB);
                    BoolMesh meshBRev = goBRev.AddComponent<BoolMesh>();
                    BoolMesh meshC    = goC.AddComponent<BoolMesh>();

                    if (meshB.Loops.Count == 0)
                    {
                        Debug.LogWarning($"[RCWB Boolean] '{objA.gameObject.name}' 减 '{objB.gameObject.name}'：B 无有效轮廓，跳过此步。");
                        continue;
                    }

                    // 膨胀 B → GetReverse → Intersection
                    InflateMesh(meshB, offsetEpsilon, miterLimit);
                    BoolMesh.GetReverse(meshB, ref meshBRev);
                    BooleanOperation.Intersection(ref meshCurrentA, ref meshBRev, ref meshC);

                    // 结果判断：
                    //   Intersection(A, ¬B) 在 A 与 B 无交叠时，可能因 ¬B（CW 环）的
                    //   射线投票不稳定而错误地返回空 meshC。
                    //   需区分两种空结果情形：
                    //     (a) A 在 B 外（两者无交叠）→ 减法无效，保持 meshCurrentA 不变
                    //     (b) A 在 B 内（A 被 B 完全覆盖）→ 正确地变空，继续传播
                    if (meshC.Loops.Count == 0 && meshCurrentA.Loops.Count > 0)
                    {
                        // 用射线检测判断 A 的某个顶点是否在原始 B（未反向）内部
                        Vector2 testPoint = meshCurrentA.Loops[0].Vertices[0].Point;
                        bool aInsideB = IsPointInsideMesh(testPoint, meshB);

                        if (!aInsideB)
                        {
                            // 情形 (a)：A 与 B 无交叠，减法对 A 无影响，静默跳过
                            Debug.Log($"[RCWB Boolean] '{objA.gameObject.name}' 减 '{objB.gameObject.name}'：两者无交叠，跳过。");
                            continue; // meshCurrentA / goCurrentA 保持原样
                        }
                        // 情形 (b)：A 被 B 完全覆盖，结果正确为空，继续传播
                    }

                    // 正常更新：meshC 成为新的 currentA
                    // （旧 goCurrentA 仍在 tempGOs 中，finally 统一销毁）
                    goCurrentA   = goC;
                    meshCurrentA = meshC;
                }

                // 结果转回 A 的局部空间
                var newLoops = WorldMeshToProfileLoops(meshCurrentA, objA.transform);

                if (newLoops.Count == 0)
                {
                    Debug.LogWarning($"[RCWB Boolean] '{objA.gameObject.name}'：减法结果为空（可能被 B 完全覆盖），ContourProfile 未修改。");
                    return false;
                }

                // 写回
                var profile = EnsureProfile(objA);
                if (profile == null)
                {
                    Debug.LogError($"[RCWB Boolean] '{objA.gameObject.name}'：无法创建 ContourProfile（场景未保存？）。");
                    return false;
                }

                Undo.RecordObject(profile, "Boolean Subtract (A - B)");
                profile.SetLoops(newLoops);
                EditorUtility.SetDirty(profile);
                AssetDatabase.SaveAssets();

                Debug.Log($"[RCWB Boolean] '{objA.gameObject.name}'：{newLoops.Count} 个环写回 → {AssetDatabase.GetAssetPath(profile)}");
                return true;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[RCWB Boolean] '{objA.gameObject.name}' 出错：{e}");
                return false;
            }
            finally
            {
                foreach (var go in tempGOs)
                    if (go != null)
                        UnityEngine.Object.DestroyImmediate(go);
            }
        }

        // ─── 点在多边形内外检测（射线法，+X 方向） ──────────────────────────
        //
        // 用于判断 A 的代表点是否在原始 B（未反向）内部，
        // 以区分"A 与 B 无交叠（减法无效）"和"A 被 B 完全覆盖（结果为空）"。
        // 注意：此处不处理退化情况（点恰好在边上），精度要求不高，用于分支判断即可。
        private static bool IsPointInsideMesh(Vector2 point, BoolMesh mesh)
        {
            int crossings = 0;
            foreach (var loop in mesh.Loops)
            {
                foreach (var edge in loop.Edges)
                {
                    float y0 = edge.VertexBegin.Point.y - point.y;
                    float y1 = edge.VertexEnd.Point.y   - point.y;

                    // 边跨过 point 所在的水平线（严格异号）
                    if ((y0 > 0f) == (y1 > 0f)) continue;

                    // 射线与边的 X 交点
                    float t  = -y0 / (y1 - y0);
                    float xCross = edge.VertexBegin.Point.x
                                 + t * (edge.VertexEnd.Point.x - edge.VertexBegin.Point.x);

                    if (xCross > point.x)
                        crossings++;
                }
            }
            return (crossings & 1) == 1; // 奇数 = 内部
        }

        // ─── Inflate：将 BoolMesh 向外均匀膨胀（Miter Join） ────────────────
        //
        // 前提：Loop 均为 CCW（由 RCWBContourProfileGenerator 规范保证）。
        // 向外法线 = 边方向的右手垂直：edge dir (dx,dy) → outward normal (dy,-dx)。
        //
        // Miter Join 公式：
        //   设顶点内角为 α，两侧外法线的角平分线为 bisector，
        //   则 bisector 与任一侧外法线的夹角 = (π-α)/2，
        //   因此  dot(bisector, n_prev) = cos((π-α)/2) = sin(α/2)。
        //   要使膨胀后对两侧边的垂直间距均等于 ε：
        //       d = ε / sin(α/2) = ε / dot(bisector, n_prev)
        //
        // miterLimit 限制 d_max = ε × miterLimit，
        //   等价于 sin(α/2)_min = 1 / miterLimit，
        //   防止极尖顶角（α → 0）产生过大的尖刺。
        private static void InflateMesh(BoolMesh mesh, float epsilon, float miterLimit)
        {
            if (epsilon <= 0f) return;

            // sin(α/2) 的最小允许值，低于此则 clamp（对应 d_max = ε * miterLimit）
            float minSinHalfAngle = 1f / Mathf.Max(miterLimit, 1f);

            foreach (var loop in mesh.Loops)
            {
                int n = loop.Vertices.Count;
                if (n < 3) continue;

                // 预先计算所有偏移量，避免修改顶点后影响后续顶点的计算
                Vector2[] offsets = new Vector2[n];

                for (int i = 0; i < n; i++)
                {
                    Vertex vPrev = loop.Vertices[(i - 1 + n) % n];
                    Vertex vCurr = loop.Vertices[i];
                    Vertex vNext = loop.Vertices[(i + 1) % n];

                    // 进入当前顶点的边方向 p（前一顶点 → 当前顶点）
                    Vector2 p = (vCurr.Point - vPrev.Point).normalized;
                    // 离开当前顶点的边方向 q（当前顶点 → 下一顶点）
                    Vector2 q = (vNext.Point - vCurr.Point).normalized;

                    // CCW 多边形，向外法线 = 右手垂直 (dy, -dx)
                    Vector2 nPrev = new Vector2( p.y, -p.x);
                    Vector2 nNext = new Vector2( q.y, -q.x);

                    // 两侧外法线的角平分线
                    Vector2 bisectorSum = nPrev + nNext;
                    if (bisectorSum.sqrMagnitude < 1e-6f)
                    {
                        // 退化：两条边近似反向（折回尖刺），两外法线相消。
                        // 直接以前一法线方向偏移 ε。
                        offsets[i] = nPrev * epsilon;
                        continue;
                    }
                    Vector2 bisector = bisectorSum.normalized;

                    // sin(α/2) = dot(bisector, nPrev)，clamp 防止极尖顶角过度膨胀
                    float sinHalfAngle = Mathf.Max(Vector2.Dot(bisector, nPrev), minSinHalfAngle);

                    offsets[i] = bisector * (epsilon / sinHalfAngle);
                }

                // 统一应用偏移（in-place）
                for (int i = 0; i < n; i++)
                    loop.Vertices[i].Point += offsets[i];
            }
        }

        // ─── ContourProfile / Sprite → 世界空间 BoolMesh ───────────────────

        private static BoolMesh BuildWorldMesh(RCWBObject rcwb, GameObject container)
        {
            var mesh = container.AddComponent<BoolMesh>();
            Transform tf = rcwb.transform;

            if (rcwb.ContourProfile != null && rcwb.ContourProfile.IsValid())
            {
                foreach (var loopData in rcwb.ContourProfile.Loops)
                {
                    if (!loopData.IsValid()) continue;
                    var loop = new Loop();
                    foreach (var pt in loopData.PointsLocal)
                    {
                        Vector3 w = tf.TransformPoint(pt.x, pt.y, 0f);
                        loop.AddVertex(new Vertex(w.x, w.y));
                    }
                    loop.GenerateEdges();
                    mesh.AddLoop(loop);
                }
            }
            else
            {
                // fallback：Sprite 物理形状（已是局部单位空间，与 ContourProfile 一致）
                var sr = rcwb.GetComponent<SpriteRenderer>();
                if (sr != null && sr.sprite != null)
                {
                    var pts = new List<Vector2>();
                    int shapeCount = sr.sprite.GetPhysicsShapeCount();
                    for (int s = 0; s < shapeCount; s++)
                    {
                        pts.Clear();
                        sr.sprite.GetPhysicsShape(s, pts);
                        if (pts.Count < 3) continue;
                        var loop = new Loop();
                        foreach (var pt in pts)
                        {
                            Vector3 w = tf.TransformPoint(pt.x, pt.y, 0f);
                            loop.AddVertex(new Vertex(w.x, w.y));
                        }
                        loop.GenerateEdges();
                        mesh.AddLoop(loop);
                    }
                }
            }

            return mesh;
        }

        // ─── 世界空间 BoolMesh → ContourLoopData（A 局部空间）───────────────

        private static List<ContourLoopData> WorldMeshToProfileLoops(
            BoolMesh mesh, Transform transformA)
        {
            var result = new List<ContourLoopData>();
            foreach (var loop in mesh.Loops)
            {
                if (loop.Vertices.Count < 3) continue;
                var localPts = new List<Vector2>();
                foreach (var vert in loop.Vertices)
                {
                    Vector3 local = transformA.InverseTransformPoint(
                        vert.Point.x, vert.Point.y, 0f);
                    localPts.Add(new Vector2(local.x, local.y));
                }
                var loopData = new ContourLoopData();
                loopData.SetData(localPts, true);
                result.Add(loopData);
            }
            return result;
        }

        // ─── 确保 A 有 ContourProfile（无则按路径约定创建）─────────────────

        private static RCWBContourProfile EnsureProfile(RCWBObject obj)
        {
            if (obj.ContourProfile != null)
                return obj.ContourProfile;

            string path = GetProfileAssetPath(obj);
            if (path == null) return null;

            CreateFolderRecursive(Path.GetDirectoryName(path).Replace('\\', '/'));

            var profile = ScriptableObject.CreateInstance<RCWBContourProfile>();
            AssetDatabase.CreateAsset(profile, path);
            AssetDatabase.SaveAssets();

            Undo.RecordObject(obj, "Assign ContourProfile (Boolean)");
            obj.ContourProfile = profile;
            EditorUtility.SetDirty(obj);

            return profile;
        }

        // 路径约定与 RCWBContourProfileGenerator 保持一致
        private static string GetProfileAssetPath(RCWBObject obj)
        {
            var scene = obj.gameObject.scene;
            if (string.IsNullOrEmpty(scene.path)) return null;

            string sceneDir  = Path.GetDirectoryName(scene.path).Replace('\\', '/');
            string sceneName = Path.GetFileNameWithoutExtension(scene.path);
            string goName    = obj.gameObject.name;

            var settings = PolygonManagerSettings.Instance;
            if (settings != null && !string.IsNullOrEmpty(settings.defaultProfileFolder))
                return $"{settings.defaultProfileFolder}/{sceneName}/{goName}_ContourProfile.asset";
            else
                return $"{sceneDir}/{sceneName}_RCWBProfiles/{goName}_ContourProfile.asset";
        }

        private static void CreateFolderRecursive(string path)
        {
            if (string.IsNullOrEmpty(path) || AssetDatabase.IsValidFolder(path)) return;
            string[] parts = path.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }
    }
}

#endif
