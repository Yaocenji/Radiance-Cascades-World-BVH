#if UNITY_EDITOR

using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using RadianceCascadesWorldBVH;

using BoolMesh = global::Mesh;

namespace RadianceCascadesWorldBVH.Editor
{
    /// <summary>
    /// 场景轮廓一键重建工具：
    ///   1. 对列表中每个 RCWBObject 重新从 Sprite 生成干净的 ContourProfile；
    ///   2. 按列表顺序依次裁剪——每个物体减去排在它前面的所有物体，
    ///      使最终所有轮廓彼此不重叠。
    ///
    /// 优先级语义：列表靠前的物体"赢得"重叠区域，靠后的物体被裁掉该区域。
    /// 可在列表中拖拽调整顺序。
    /// </summary>
    public class RCWBSceneContourRebuildTool : EditorWindow
    {
        // ─── 物体列表 ────────────────────────────────────────────────────────
        private List<RCWBObject> m_Objects = new List<RCWBObject>();

        // ─── 膨胀参数（与 BooleanSubtractTool 保持一致的默认值）────────────
        private float m_OffsetEpsilon = 0.005f;
        private float m_MiterLimit    = 4f;

        // ─── Alpha 阈值（轮廓生成用）────────────────────────────────────────
        private float m_AlphaThreshold = 0.01f;

        // ─── UI 状态 ─────────────────────────────────────────────────────────
        private Vector2 m_Scroll;
        private int     m_DragFromIndex = -1;

        [MenuItem("Tools/RCWB/Scene Contour Rebuild")]
        public static void ShowWindow()
        {
            GetWindow<RCWBSceneContourRebuildTool>("Scene Contour Rebuild");
        }

        private void OnEnable()
        {
            if (m_Objects == null) m_Objects = new List<RCWBObject>();
        }

        // ─── UI ──────────────────────────────────────────────────────────────

        private void OnGUI()
        {
            GUILayout.Label("RCWB 场景轮廓一键重建", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "步骤：\n" +
                "  1. 对列表中每个物体重新从 Sprite 生成干净轮廓\n" +
                "  2. 按列表顺序互相裁剪，消除所有重叠区域\n\n" +
                "优先级：列表靠上的物体优先，靠下的物体被裁去与上方物体的重叠部分。\n" +
                "可拖拽列表行调整顺序。",
                MessageType.None);

            EditorGUILayout.Space();

            // ── 列表 ──
            EditorGUILayout.LabelField("物体列表（按优先级排列）", EditorStyles.boldLabel);
            m_Scroll = EditorGUILayout.BeginScrollView(m_Scroll, GUILayout.MaxHeight(200));
            DrawObjectList();
            EditorGUILayout.EndScrollView();

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("+ 添加空槽"))
                    m_Objects.Add(null);
                if (GUILayout.Button("从选择填入"))
                    FillFromSelection();
                if (GUILayout.Button("填入场景所有"))
                    FillAllFromScene();
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            // ── 参数 ──
            EditorGUILayout.LabelField("参数", EditorStyles.boldLabel);

            m_AlphaThreshold = EditorGUILayout.Slider(
                new GUIContent("Alpha 阈值", "生成轮廓时的像素透明度阈值。"),
                m_AlphaThreshold, 0f, 1f);

            m_OffsetEpsilon = EditorGUILayout.FloatField(
                new GUIContent("裁剪间距 Epsilon（世界单位）",
                    "裁剪前先将遮挡物轮廓膨胀此距离，保留等距细缝。典型值：0.001~0.01。"),
                m_OffsetEpsilon);
            m_OffsetEpsilon = Mathf.Max(0f, m_OffsetEpsilon);

            m_MiterLimit = EditorGUILayout.Slider(
                new GUIContent("Miter 上限", "膨胀时尖角处最大倍数，防止过大尖刺。"),
                m_MiterLimit, 1f, 20f);

            EditorGUILayout.Space();

            // ── 验证 & 执行 ──
            string error = Validate();
            if (error != null)
                EditorGUILayout.HelpBox(error, MessageType.Warning);

            EditorGUI.BeginDisabledGroup(error != null);
            int validCount = CountValid();
            if (GUILayout.Button(
                    $"一键重建：{validCount} 个物体",
                    GUILayout.Height(40)))
            {
                if (EditorUtility.DisplayDialog(
                    "确认执行",
                    $"将重建 {validCount} 个物体的 ContourProfile，此操作会覆盖现有轮廓数据。\n\n建议先保存场景。\n\n继续？",
                    "执行", "取消"))
                {
                    Execute();
                }
            }
            EditorGUI.EndDisabledGroup();
        }

        // ─── 列表绘制（支持拖拽排序）────────────────────────────────────────

        private void DrawObjectList()
        {
            if (m_Objects.Count == 0)
            {
                EditorGUILayout.LabelField("（空，请添加物体）", EditorStyles.miniLabel);
                return;
            }

            for (int i = 0; i < m_Objects.Count; i++)
            {
                var rowRect = EditorGUILayout.BeginHorizontal();

                // 序号标签（表示优先级）
                EditorGUILayout.LabelField($"{i + 1}.", GUILayout.Width(24));

                m_Objects[i] = (RCWBObject)EditorGUILayout.ObjectField(
                    m_Objects[i], typeof(RCWBObject), true);

                // 状态标签
                string status = m_Objects[i] != null
                    ? GetStatusLabel(m_Objects[i])
                    : "—";
                EditorGUILayout.LabelField(status, EditorStyles.miniLabel, GUILayout.Width(80));

                // 上移 / 下移
                EditorGUI.BeginDisabledGroup(i == 0);
                if (GUILayout.Button("↑", GUILayout.Width(22)))
                {
                    (m_Objects[i - 1], m_Objects[i]) = (m_Objects[i], m_Objects[i - 1]);
                    GUIUtility.ExitGUI();
                }
                EditorGUI.EndDisabledGroup();

                EditorGUI.BeginDisabledGroup(i == m_Objects.Count - 1);
                if (GUILayout.Button("↓", GUILayout.Width(22)))
                {
                    (m_Objects[i + 1], m_Objects[i]) = (m_Objects[i], m_Objects[i + 1]);
                    GUIUtility.ExitGUI();
                }
                EditorGUI.EndDisabledGroup();

                if (GUILayout.Button("−", GUILayout.Width(22)))
                {
                    m_Objects.RemoveAt(i);
                    GUIUtility.ExitGUI();
                }

                EditorGUILayout.EndHorizontal();
            }
        }

        private static string GetStatusLabel(RCWBObject obj)
        {
            var sr = obj.GetComponent<SpriteRenderer>();
            bool hasSprite = sr != null && sr.sprite != null;
            bool hasProfile = obj.ContourProfile != null && obj.ContourProfile.IsValid();

            if (!hasSprite) return "无Sprite";
            if (hasProfile) return $"Profile({obj.ContourProfile.LoopCount})";
            return "待生成";
        }

        // ─── 填充辅助 ────────────────────────────────────────────────────────

        private void FillFromSelection()
        {
            foreach (var go in Selection.gameObjects)
                foreach (var r in go.GetComponentsInChildren<RCWBObject>(true))
                    if (!m_Objects.Contains(r))
                        m_Objects.Add(r);
        }

        private void FillAllFromScene()
        {
            var all = Object.FindObjectsOfType<RCWBObject>();
            foreach (var obj in all)
                if (!m_Objects.Contains(obj))
                    m_Objects.Add(obj);
        }

        // ─── 验证 ────────────────────────────────────────────────────────────

        private string Validate()
        {
            if (CountValid() < 2)
                return "至少需要 2 个有效物体。";

            foreach (var obj in m_Objects)
            {
                if (obj == null) continue;
                var sr = obj.GetComponent<SpriteRenderer>();
                if (sr == null || sr.sprite == null)
                    return $"'{obj.gameObject.name}' 没有 SpriteRenderer 或 Sprite。";

                if (string.IsNullOrEmpty(obj.gameObject.scene.path))
                    return "场景尚未保存，请先保存场景。";
            }
            return null;
        }

        private int CountValid()
        {
            int n = 0;
            foreach (var o in m_Objects) if (o != null) n++;
            return n;
        }

        // ─── 核心执行 ────────────────────────────────────────────────────────

        private void Execute()
        {
            // 收集有效物体
            var valid = new List<RCWBObject>();
            foreach (var o in m_Objects) if (o != null) valid.Add(o);

            int total = valid.Count;

            try
            {
                // ── 阶段 1：重新生成所有轮廓 ──
                // 存储干净的世界空间 BoolMesh，后续裁剪用
                // key: 物体索引，value: 对应的干净 BoolMesh（临时 GO）
                var tempGOs         = new List<GameObject>();
                var cleanWorldMeshes = new List<BoolMesh>(); // 与 valid 一一对应

                for (int i = 0; i < total; i++)
                {
                    EditorUtility.DisplayProgressBar(
                        "RCWB 场景轮廓重建",
                        $"[1/2] 生成轮廓 {valid[i].gameObject.name} ({i + 1}/{total})",
                        (float)i / (total * 2));

                    var go   = new GameObject("__RebuildTemp") { hideFlags = HideFlags.HideAndDontSave };
                    tempGOs.Add(go);

                    BoolMesh mesh = BuildFreshWorldMesh(valid[i], go, m_AlphaThreshold);
                    cleanWorldMeshes.Add(mesh);

                    if (mesh == null || mesh.Loops.Count == 0)
                        Debug.LogWarning($"[RCWB Rebuild] '{valid[i].gameObject.name}'：无法生成有效轮廓，将跳过裁剪步骤。");
                }

                // ── 阶段 2：按优先级互相裁剪 ──
                // currentMeshes[i] 存储第 i 个物体当前最新的轮廓（随裁剪更新）
                // 这里用独立的 GO 列表管理，避免与 cleanWorldMeshes 混用
                var currentGOs    = new List<GameObject>();
                var currentMeshes = new List<BoolMesh>();

                for (int i = 0; i < total; i++)
                {
                    // 先克隆一份干净 mesh 作为 currentMesh[i] 的起点
                    var goC = new GameObject("__RebuildCurrent") { hideFlags = HideFlags.HideAndDontSave };
                    tempGOs.Add(goC);
                    BoolMesh mc = goC.AddComponent<BoolMesh>();
                    CopyLoops(cleanWorldMeshes[i], mc);
                    currentGOs.Add(goC);
                    currentMeshes.Add(mc);
                }

                int successCount = 0;
                int failCount    = 0;

                for (int i = 0; i < total; i++)
                {
                    EditorUtility.DisplayProgressBar(
                        "RCWB 场景轮廓重建",
                        $"[2/2] 裁剪 {valid[i].gameObject.name} ({i + 1}/{total})",
                        0.5f + (float)i / (total * 2));

                    // 物体 i 减去排在它前面（0 ~ i-1）且轮廓有效的所有物体
                    bool ok = SubtractPredecessors(
                        valid[i], i, currentMeshes, tempGOs,
                        m_OffsetEpsilon, m_MiterLimit);

                    if (ok) successCount++;
                    else    failCount++;
                }

                EditorUtility.ClearProgressBar();

                // ── 批量 Save ──
                AssetDatabase.SaveAssets();

                string summary = $"[RCWB Rebuild] 完成：{successCount} 个物体成功";
                if (failCount > 0) summary += $"，{failCount} 个失败（详见 Console）";
                Debug.Log(summary);

                if (failCount > 0)
                    EditorUtility.DisplayDialog("部分失败",
                        $"成功 {successCount} 个，失败 {failCount} 个。\n详见 Console。", "确定");

                // 统一销毁所有临时 GO
                foreach (var go in tempGOs)
                    if (go != null)
                        Object.DestroyImmediate(go);
            }
            catch (System.Exception e)
            {
                EditorUtility.ClearProgressBar();
                Debug.LogError($"[RCWB Rebuild] 执行中断：{e}");
                EditorUtility.DisplayDialog("执行出错", e.Message, "确定");
            }
        }

        /// <summary>
        /// 对物体 valid[index]，依次减去 currentMeshes[0..index-1]（优先级更高的物体）。
        /// 减法结果写回 currentMeshes[index] 并保存到 ContourProfile。
        /// </summary>
        private static bool SubtractPredecessors(
            RCWBObject obj, int index,
            List<BoolMesh> currentMeshes,
            List<GameObject> tempGOs,
            float epsilon, float miterLimit)
        {
            try
            {
                BoolMesh meshCurrent = currentMeshes[index];

                if (meshCurrent == null || meshCurrent.Loops.Count == 0)
                {
                    Debug.LogWarning($"[RCWB Rebuild] 跳过 '{obj.gameObject.name}'：无有效轮廓。");
                    return false;
                }

                // 依次减去每个前驱物体（优先级更高）
                for (int j = 0; j < index; j++)
                {
                    BoolMesh meshB = currentMeshes[j];
                    if (meshB == null || meshB.Loops.Count == 0) continue;

                    // 克隆膨胀用的 B（不影响原始 currentMeshes[j]）
                    var goInflated = new GameObject("__Inflate") { hideFlags = HideFlags.HideAndDontSave };
                    tempGOs.Add(goInflated);
                    BoolMesh meshBInflated = goInflated.AddComponent<BoolMesh>();
                    CopyLoops(meshB, meshBInflated);

                    var goBRev = new GameObject("__BRev") { hideFlags = HideFlags.HideAndDontSave };
                    var goResult = new GameObject("__Result") { hideFlags = HideFlags.HideAndDontSave };
                    tempGOs.Add(goBRev);
                    tempGOs.Add(goResult);

                    BoolMesh meshBRev   = goBRev.AddComponent<BoolMesh>();
                    BoolMesh meshResult = goResult.AddComponent<BoolMesh>();

                    InflateMesh(meshBInflated, epsilon, miterLimit);
                    BoolMesh.GetReverse(meshBInflated, ref meshBRev);
                    BooleanOperation.Intersection(ref meshCurrent, ref meshBRev, ref meshResult);

                    // 空结果分支处理（与 BooleanSubtractTool 逻辑一致）
                    if (meshResult.Loops.Count == 0 && meshCurrent.Loops.Count > 0)
                    {
                        Vector2 testPt = meshCurrent.Loops[0].Vertices[0].Point;
                        if (!IsPointInsideMesh(testPt, meshBInflated))
                        {
                            // A 与 B 无交叠，跳过
                            continue;
                        }
                        // A 被 B 完全覆盖，结果为空，继续传播
                    }

                    meshCurrent = meshResult;

                    if (meshCurrent.Loops.Count == 0) break; // 已完全被覆盖
                }

                // 更新列表中的引用
                currentMeshes[index] = meshCurrent;

                // 结果转回局部空间并写回
                var newLoops = WorldMeshToProfileLoops(meshCurrent, obj.transform);

                if (newLoops.Count == 0)
                {
                    Debug.LogWarning($"[RCWB Rebuild] '{obj.gameObject.name}'：裁剪结果为空（被前驱物体完全覆盖），ContourProfile 未修改。");
                    return false;
                }

                var profile = EnsureProfile(obj);
                if (profile == null)
                {
                    Debug.LogError($"[RCWB Rebuild] '{obj.gameObject.name}'：无法创建 ContourProfile。");
                    return false;
                }

                Undo.RecordObject(profile, "Scene Contour Rebuild");
                profile.SetLoops(newLoops);
                EditorUtility.SetDirty(profile);

                Debug.Log($"[RCWB Rebuild] '{obj.gameObject.name}'：{newLoops.Count} 个环写回 → {AssetDatabase.GetAssetPath(profile)}");
                return true;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[RCWB Rebuild] '{obj.gameObject.name}' 裁剪出错：{e}");
                return false;
            }
        }

        // ─── 从 Sprite 生成世界空间 BoolMesh（不依赖现有 ContourProfile）──────

        private static BoolMesh BuildFreshWorldMesh(
            RCWBObject rcwb, GameObject container, float alphaThreshold)
        {
            var mesh = container.AddComponent<BoolMesh>();
            var sr   = rcwb.GetComponent<SpriteRenderer>();
            if (sr == null || sr.sprite == null) return mesh;

            Sprite sprite   = sr.sprite;
            string path     = AssetDatabase.GetAssetPath(sprite);
            var importer    = AssetImporter.GetAtPath(path) as TextureImporter;
            var sourceTex   = AssetDatabase.LoadAssetAtPath<Texture2D>(path);

            if (importer == null || sourceTex == null || !sourceTex.isReadable)
            {
                Debug.LogWarning($"[RCWB Rebuild] '{rcwb.gameObject.name}'：纹理不可读或无 Importer，跳过轮廓生成。");
                return mesh;
            }

            Rect sourceRect = GetSourceRect(sprite, importer);
            int x0 = Mathf.FloorToInt(sourceRect.x);
            int y0 = Mathf.FloorToInt(sourceRect.y);
            int w  = Mathf.FloorToInt(sourceRect.width);
            int h  = Mathf.FloorToInt(sourceRect.height);

            Color[] pixels = sourceTex.GetPixels(x0, y0, w, h);
            bool[,] mask   = BuildMask(pixels, w, h, alphaThreshold);

            var rawLoops = TraceAllContours(mask, w, h);
            Vector2 pivot = sprite.pivot;
            float   ppu   = sprite.pixelsPerUnit;
            Transform tf  = rcwb.transform;

            foreach (var rawLoop in rawLoops)
            {
                var simplified = RemoveCollinearPoints(rawLoop);
                if (simplified.Count < 3) continue;

                var loop = new Loop();
                foreach (var gp in simplified)
                {
                    float lx = (gp.x - pivot.x) / ppu;
                    float ly = (gp.y - pivot.y) / ppu;
                    Vector3 w3 = tf.TransformPoint(lx, ly, 0f);
                    loop.AddVertex(new Vertex(w3.x, w3.y));
                }
                loop.GenerateEdges();
                mesh.AddLoop(loop);
            }

            return mesh;
        }

        // ─── 辅助：拷贝 Loop 到另一个 BoolMesh ─────────────────────────────

        private static void CopyLoops(BoolMesh src, BoolMesh dst)
        {
            if (src == null) return;
            foreach (var srcLoop in src.Loops)
            {
                var loop = new Loop();
                foreach (var v in srcLoop.Vertices)
                    loop.AddVertex(new Vertex(v.Point.x, v.Point.y));
                loop.GenerateEdges();
                dst.AddLoop(loop);
            }
        }

        // ─── 世界空间 BoolMesh → ContourLoopData（局部空间）────────────────

        private static List<ContourLoopData> WorldMeshToProfileLoops(
            BoolMesh mesh, Transform tf)
        {
            var result = new List<ContourLoopData>();
            foreach (var loop in mesh.Loops)
            {
                if (loop.Vertices.Count < 3) continue;
                var localPts = new List<Vector2>();
                foreach (var vert in loop.Vertices)
                {
                    Vector3 local = tf.InverseTransformPoint(vert.Point.x, vert.Point.y, 0f);
                    localPts.Add(new Vector2(local.x, local.y));
                }
                var loopData = new ContourLoopData();
                loopData.SetData(localPts, true);
                result.Add(loopData);
            }
            return result;
        }

        // ─── 确保 ContourProfile 存在 ────────────────────────────────────────

        private static RCWBContourProfile EnsureProfile(RCWBObject obj)
        {
            if (obj.ContourProfile != null) return obj.ContourProfile;

            string assetPath = GetProfileAssetPath(obj);
            if (assetPath == null) return null;

            CreateFolderRecursive(Path.GetDirectoryName(assetPath).Replace('\\', '/'));

            var profile = ScriptableObject.CreateInstance<RCWBContourProfile>();
            AssetDatabase.CreateAsset(profile, assetPath);

            Undo.RecordObject(obj, "Assign ContourProfile (Rebuild)");
            obj.ContourProfile = profile;
            EditorUtility.SetDirty(obj);

            return profile;
        }

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
            string[] parts  = path.Split('/');
            string   current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }

        // ─── 膨胀（与 BooleanSubtractTool.InflateMesh 完全一致）─────────────

        private static void InflateMesh(BoolMesh mesh, float epsilon, float miterLimit)
        {
            if (epsilon <= 0f) return;
            float minSinHalfAngle = 1f / Mathf.Max(miterLimit, 1f);

            foreach (var loop in mesh.Loops)
            {
                int n = loop.Vertices.Count;
                if (n < 3) continue;

                Vector2[] offsets = new Vector2[n];
                for (int i = 0; i < n; i++)
                {
                    Vertex vPrev = loop.Vertices[(i - 1 + n) % n];
                    Vertex vCurr = loop.Vertices[i];
                    Vertex vNext = loop.Vertices[(i + 1) % n];

                    Vector2 p = (vCurr.Point - vPrev.Point).normalized;
                    Vector2 q = (vNext.Point - vCurr.Point).normalized;

                    Vector2 nPrev = new Vector2( p.y, -p.x);
                    Vector2 nNext = new Vector2( q.y, -q.x);

                    Vector2 bisectorSum = nPrev + nNext;
                    if (bisectorSum.sqrMagnitude < 1e-6f)
                    {
                        offsets[i] = nPrev * epsilon;
                        continue;
                    }
                    Vector2 bisector = bisectorSum.normalized;
                    float sinHalfAngle = Mathf.Max(Vector2.Dot(bisector, nPrev), minSinHalfAngle);
                    offsets[i] = bisector * (epsilon / sinHalfAngle);
                }
                for (int i = 0; i < n; i++)
                    loop.Vertices[i].Point += offsets[i];
            }
        }

        // ─── 点在多边形内外检测（射线法）────────────────────────────────────

        private static bool IsPointInsideMesh(Vector2 point, BoolMesh mesh)
        {
            int crossings = 0;
            foreach (var loop in mesh.Loops)
            {
                foreach (var edge in loop.Edges)
                {
                    float y0 = edge.VertexBegin.Point.y - point.y;
                    float y1 = edge.VertexEnd.Point.y   - point.y;
                    if ((y0 > 0f) == (y1 > 0f)) continue;
                    float t      = -y0 / (y1 - y0);
                    float xCross = edge.VertexBegin.Point.x
                                 + t * (edge.VertexEnd.Point.x - edge.VertexBegin.Point.x);
                    if (xCross > point.x) crossings++;
                }
            }
            return (crossings & 1) == 1;
        }

        // ─── 轮廓追踪（移植自 RCWBContourProfileGenerator）──────────────────

        private static Rect GetSourceRect(Sprite sprite, TextureImporter importer)
        {
            if (importer.spriteImportMode == SpriteImportMode.Multiple)
            {
                foreach (var sheet in importer.spritesheet)
                    if (sheet.name == sprite.name) return sheet.rect;
            }
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(AssetDatabase.GetAssetPath(sprite));
            return new Rect(0, 0, tex.width, tex.height);
        }

        private static bool[,] BuildMask(Color[] pixels, int w, int h, float threshold)
        {
            var mask = new bool[w, h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    mask[x, y] = pixels[y * w + x].a > threshold;
            return mask;
        }

        private enum Dir { Right = 0, Up = 1, Left = 2, Down = 3 }
        private static readonly int[] s_Dx = { 1, 0, -1, 0 };
        private static readonly int[] s_Dy = { 0, 1, 0, -1 };

        private static List<List<Vector2Int>> TraceAllContours(bool[,] mask, int w, int h)
        {
            var edgeVisited = new bool[w + 1, h + 1, 4];
            var allLoops    = new List<List<Vector2Int>>();

            for (int gy = 0; gy <= h; gy++)
            for (int gx = 0; gx <= w; gx++)
            for (int d  = 0; d  < 4; d++)
            {
                if (edgeVisited[gx, gy, d]) continue;
                if (!IsBoundaryEdge(mask, w, h, gx, gy, (Dir)d)) continue;
                var loop = TraceSingleContour(mask, w, h, edgeVisited, gx, gy, (Dir)d);
                if (loop != null && loop.Count >= 4)
                    allLoops.Add(loop);
            }
            return allLoops;
        }

        private static bool IsBoundaryEdge(bool[,] mask, int w, int h, int gx, int gy, Dir dir)
        {
            GetEdgeNeighbors(gx, gy, dir, out int lx, out int ly, out int rx, out int ry);
            bool lSolid = InBounds(lx, ly, w, h) && mask[lx, ly];
            bool rSolid = InBounds(rx, ry, w, h) && mask[rx, ry];
            return lSolid && !rSolid;
        }

        private static void GetEdgeNeighbors(
            int gx, int gy, Dir dir, out int lx, out int ly, out int rx, out int ry)
        {
            switch (dir)
            {
                case Dir.Right: lx = gx;     ly = gy;     rx = gx;     ry = gy - 1; break;
                case Dir.Up:    lx = gx - 1; ly = gy;     rx = gx;     ry = gy;     break;
                case Dir.Left:  lx = gx - 1; ly = gy - 1; rx = gx - 1; ry = gy;     break;
                case Dir.Down:  lx = gx;     ly = gy - 1; rx = gx - 1; ry = gy - 1; break;
                default:        lx = ly = rx = ry = 0;                               break;
            }
        }

        private static List<Vector2Int> TraceSingleContour(
            bool[,] mask, int w, int h,
            bool[,,] edgeVisited, int startGx, int startGy, Dir startDir)
        {
            var loop = new List<Vector2Int>();
            int gx   = startGx;
            int gy   = startGy;
            Dir dir  = startDir;

            int maxSteps = (w + 1) * (h + 1) * 4 + 1;
            for (int step = 0; step < maxSteps; step++)
            {
                if (edgeVisited[gx, gy, (int)dir])
                {
                    if (gx == startGx && gy == startGy && dir == startDir && loop.Count > 0) break;
                    if (loop.Count > 0) break;
                    return null;
                }
                edgeVisited[gx, gy, (int)dir] = true;
                loop.Add(new Vector2Int(gx, gy));

                int ngx = gx + s_Dx[(int)dir];
                int ngy = gy + s_Dy[(int)dir];
                dir = ChooseNextDir(mask, w, h, ngx, ngy, dir);
                gx  = ngx;
                gy  = ngy;

                if (gx == startGx && gy == startGy && dir == startDir) break;
            }
            return loop;
        }

        private static Dir ChooseNextDir(bool[,] mask, int w, int h, int gx, int gy, Dir cur)
        {
            Dir right  = (Dir)(((int)cur + 3) % 4);
            Dir straight = cur;
            Dir left   = (Dir)(((int)cur + 1) % 4);
            Dir uTurn  = (Dir)(((int)cur + 2) % 4);

            if (IsBoundaryEdge(mask, w, h, gx, gy, right))    return right;
            if (IsBoundaryEdge(mask, w, h, gx, gy, straight)) return straight;
            if (IsBoundaryEdge(mask, w, h, gx, gy, left))     return left;
            return uTurn;
        }

        private static bool InBounds(int x, int y, int w, int h)
            => x >= 0 && x < w && y >= 0 && y < h;

        private static List<Vector2Int> RemoveCollinearPoints(List<Vector2Int> loop)
        {
            if (loop.Count < 3) return loop;
            var result = new List<Vector2Int>();
            int n = loop.Count;
            for (int i = 0; i < n; i++)
            {
                Vector2Int prev = loop[(i - 1 + n) % n];
                Vector2Int curr = loop[i];
                Vector2Int next = loop[(i + 1) % n];
                if (!(prev.x == curr.x && curr.x == next.x) &&
                    !(prev.y == curr.y && curr.y == next.y))
                    result.Add(curr);
            }
            return result;
        }
    }
}

#endif
