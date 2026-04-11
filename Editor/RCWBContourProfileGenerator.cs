#if UNITY_EDITOR

using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using RadianceCascadesWorldBVH;

namespace RadianceCascadesWorldBVH.Editor
{
    /// <summary>
    /// 从场景中选中的 RCWBObject 自动生成 RCWBContourProfile。
    /// 算法与 SpritePhysicsShapeGenerator 完全一致（像素追踪 + 轮廓简化），
    /// 区别仅在于：输出坐标系为局部 Unity 单位空间（除以 PPU，以 Pivot 为原点），
    /// 并写入 RCWBContourProfile ScriptableObject 而非 Sprite 物理形状。
    /// </summary>
    public class RCWBContourProfileGenerator : EditorWindow
    {
        private RadianceCascadesWorldBVH.RCWBObject targetObject;
        private float alphaThreshold = 0.01f;

        [MenuItem("Tools/RCWB/Contour Profile Generator")]
        public static void ShowWindow()
        {
            GetWindow<RCWBContourProfileGenerator>("Contour Profile Generator");
        }

        private void OnEnable()
        {
            Selection.selectionChanged += OnSelectionChanged;
            OnSelectionChanged();
        }

        private void OnDisable()
        {
            Selection.selectionChanged -= OnSelectionChanged;
        }

        private void OnSelectionChanged()
        {
            if (Selection.activeGameObject != null)
            {
                var rcwb = Selection.activeGameObject.GetComponent<RCWBObject>();
                if (rcwb != null)
                {
                    targetObject = rcwb;
                    Repaint();
                }
            }
        }

        private void OnGUI()
        {
            GUILayout.Label("RCWB Contour Profile 生成器", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            targetObject = (RCWBObject)EditorGUILayout.ObjectField(
                "目标 RCWBObject", targetObject, typeof(RCWBObject), true);

            EditorGUILayout.Space();
            alphaThreshold = EditorGUILayout.Slider("Alpha 阈值", alphaThreshold, 0f, 1f);
            EditorGUILayout.Space();

            // 预览输出路径
            if (targetObject != null)
            {
                string path = GetTargetAssetPath(targetObject);
                if (path != null)
                    EditorGUILayout.HelpBox($"将保存到：\n{path}", MessageType.None);
                else
                    EditorGUILayout.HelpBox("场景尚未保存，请先保存场景再生成。", MessageType.Warning);
            }

            EditorGUILayout.Space();

            // 单个生成
            EditorGUI.BeginDisabledGroup(targetObject == null);
            if (GUILayout.Button("生成 ContourProfile", GUILayout.Height(30)))
                GenerateProfile(targetObject);
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            // 批量生成
            if (GUILayout.Button("批量生成（场景所有合法 RCWBObject）", GUILayout.Height(30)))
                GenerateAllProfiles();

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "工作流程：\n" +
                "1. 从源纹理（非 Atlas）读取 Sprite 像素\n" +
                "2. 按 Alpha 阈值二值化\n" +
                "3. 追踪像素边界轮廓（沿像素格点走）\n" +
                "4. 去除共线冗余顶点\n" +
                "5. 转换为局部 Unity 单位空间：(gridPoint − pivot) / PPU\n" +
                "6. 保存 RCWBContourProfile 资产，并赋值到 RCWBObject.ContourProfile",
                MessageType.Info);
        }

        // ─────────────────────────────────────────────────────────────────
        //  批量生成
        // ─────────────────────────────────────────────────────────────────

        private void GenerateAllProfiles()
        {
            var valid = Object.FindObjectsOfType<RCWBObject>()
                .Where(o => o.IsWall && o.GetComponent<SpriteRenderer>()?.sprite != null)
                .ToList();

            if (valid.Count == 0)
            {
                EditorUtility.DisplayDialog("提示", "场景中没有找到合法的 RCWBObject\n（IsWall=true 且具有 Sprite）。", "确定");
                return;
            }

            int success = 0;
            for (int i = 0; i < valid.Count; i++)
            {
                EditorUtility.DisplayProgressBar(
                    "批量生成 ContourProfile",
                    valid[i].gameObject.name,
                    (float)i / valid.Count);
                try
                {
                    GenerateProfile(valid[i], silent: true);
                    success++;
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[RCWB] 生成 '{valid[i].gameObject.name}' 时出错：{e.Message}");
                }
            }

            EditorUtility.ClearProgressBar();
            Debug.Log($"[RCWB] 批量生成完成：{success}/{valid.Count} 个物体成功。");
        }

        // ─────────────────────────────────────────────────────────────────
        //  生成入口
        // ─────────────────────────────────────────────────────────────────

        private void GenerateProfile(RCWBObject obj, bool silent = false)
        {
            // 1. 检查 SpriteRenderer
            var sr = obj.GetComponent<SpriteRenderer>();
            if (sr == null || sr.sprite == null)
            {
                if (silent) { Debug.LogWarning($"[RCWB] 跳过 '{obj.gameObject.name}'：没有有效的 SpriteRenderer 或 Sprite。"); return; }
                EditorUtility.DisplayDialog("错误", "目标对象没有有效的 SpriteRenderer 或 Sprite。", "确定");
                return;
            }

            Sprite sprite = sr.sprite;

            // 2. 获取源纹理与 importer
            string spritePath = AssetDatabase.GetAssetPath(sprite);
            var importer = AssetImporter.GetAtPath(spritePath) as TextureImporter;
            if (importer == null)
            {
                if (silent) { Debug.LogWarning($"[RCWB] 跳过 '{obj.gameObject.name}'：无法获取 TextureImporter。"); return; }
                EditorUtility.DisplayDialog("错误", "无法获取 TextureImporter。", "确定");
                return;
            }

            var sourceTex = AssetDatabase.LoadAssetAtPath<Texture2D>(spritePath);
            if (sourceTex == null)
            {
                if (silent) { Debug.LogWarning($"[RCWB] 跳过 '{obj.gameObject.name}'：无法加载源纹理 {spritePath}。"); return; }
                EditorUtility.DisplayDialog("错误", $"无法加载源纹理：{spritePath}", "确定");
                return;
            }
            if (!sourceTex.isReadable)
            {
                if (silent) { Debug.LogWarning($"[RCWB] 跳过 '{obj.gameObject.name}'：纹理 '{sourceTex.name}' 不可读，请勾选 Read/Write。"); return; }
                EditorUtility.DisplayDialog("错误",
                    $"纹理 '{sourceTex.name}' 不可读。请在 Import Settings 中勾选 Read/Write。", "确定");
                return;
            }

            // 3. 确认目标路径（需要场景已保存）
            string targetPath = GetTargetAssetPath(obj);
            if (targetPath == null)
            {
                if (silent) { Debug.LogWarning($"[RCWB] 跳过 '{obj.gameObject.name}'：场景尚未保存。"); return; }
                EditorUtility.DisplayDialog("错误", "场景尚未保存，请先保存场景。", "确定");
                return;
            }

            // 4. 若文件已存在：静默时直接覆盖，否则询问
            if (!silent && File.Exists(Path.GetFullPath(targetPath)))
            {
                int choice = EditorUtility.DisplayDialogComplex(
                    "Profile 已存在",
                    $"文件已存在：\n{targetPath}\n\n是否覆盖？",
                    "覆盖", "取消", "使用已有");

                if (choice == 1) return;

                if (choice == 2)
                {
                    var existing = AssetDatabase.LoadAssetAtPath<RCWBContourProfile>(targetPath);
                    if (existing != null)
                    {
                        Undo.RecordObject(obj, "Assign Existing ContourProfile");
                        obj.ContourProfile = existing;
                        EditorUtility.SetDirty(obj);
                        Debug.Log($"[RCWB] 已关联现有 Profile：{targetPath}");
                    }
                    return;
                }
            }

            // 5. 读像素，追踪轮廓
            Rect sourceRect = GetSourceRect(sprite, importer);
            int x0 = Mathf.FloorToInt(sourceRect.x);
            int y0 = Mathf.FloorToInt(sourceRect.y);
            int w  = Mathf.FloorToInt(sourceRect.width);
            int h  = Mathf.FloorToInt(sourceRect.height);

            Color[] pixels = sourceTex.GetPixels(x0, y0, w, h);
            bool[,] mask   = BuildMask(pixels, w, h);

            List<List<Vector2Int>> rawLoops = TraceAllContours(mask, w, h);
            if (rawLoops.Count == 0)
            {
                if (silent) { Debug.LogWarning($"[RCWB] 跳过 '{obj.gameObject.name}'：未检测到任何轮廓，请检查 Alpha 阈值设置。"); return; }
                EditorUtility.DisplayDialog("警告", "未检测到任何轮廓，请检查 Alpha 阈值设置。", "确定");
                return;
            }

            // 6. 简化并转换坐标
            // sprite.pivot：以 Sprite 在源图中的左下角为原点，单位为像素
            // 与 GetPixels(x0, y0, ...) 读出的 grid 坐标系一致，可直接相减
            Vector2 pivot = sprite.pivot;
            float   ppu   = sprite.pixelsPerUnit;

            var loopDataList = new List<ContourLoopData>();
            foreach (var rawLoop in rawLoops)
            {
                var simplified = RemoveCollinearPoints(rawLoop);
                if (simplified.Count < 3) continue;

                var localUnits = ConvertToLocalUnits(simplified, pivot, ppu);
                var loopData   = new ContourLoopData();
                loopData.SetData(localUnits, true);
                loopDataList.Add(loopData);
            }

            if (loopDataList.Count == 0)
            {
                if (silent) { Debug.LogWarning($"[RCWB] 跳过 '{obj.gameObject.name}'：简化后无有效轮廓。"); return; }
                EditorUtility.DisplayDialog("警告", "简化后无有效轮廓。", "确定");
                return;
            }

            // 7. 创建或覆盖 ScriptableObject 资产
            CreateFolderRecursive(Path.GetDirectoryName(targetPath).Replace('\\', '/'));

            RCWBContourProfile profile = AssetDatabase.LoadAssetAtPath<RCWBContourProfile>(targetPath);
            if (profile == null)
            {
                profile = CreateInstance<RCWBContourProfile>();
                profile.SetLoops(loopDataList);
                AssetDatabase.CreateAsset(profile, targetPath);
            }
            else
            {
                profile.SetLoops(loopDataList);
                EditorUtility.SetDirty(profile);
            }
            AssetDatabase.SaveAssets();

            // 8. 赋值到 RCWBObject
            Undo.RecordObject(obj, "Generate ContourProfile");
            obj.ContourProfile = profile;
            EditorUtility.SetDirty(obj);

            int totalVerts = 0;
            foreach (var l in loopDataList) totalVerts += l.PointCount;
            Debug.Log($"[RCWB] ContourProfile 生成完毕：{loopDataList.Count} 个轮廓，" +
                      $"共 {totalVerts} 个顶点 → {targetPath}");
        }

        // ─────────────────────────────────────────────────────────────────
        //  坐标转换
        // ─────────────────────────────────────────────────────────────────

        /// <summary>
        /// 将格点坐标转换为局部 Unity 单位空间。
        /// 格点 (0,0) = 源 rect 左下角 = Sprite 原始左下角（两者一致，无需额外偏移）。
        /// pivot 以相同原点为基准（sprite.pivot 的定义）。
        /// 结果与 Sprite.GetPhysicsShape() 返回的坐标系完全一致，
        /// 可直接送入 transform.TransformPoint() 得到世界坐标。
        /// </summary>
        private static List<Vector2> ConvertToLocalUnits(
            List<Vector2Int> gridPoints, Vector2 pivot, float ppu)
        {
            var result = new List<Vector2>(gridPoints.Count);
            foreach (var gp in gridPoints)
            {
                result.Add(new Vector2(
                    (gp.x - pivot.x) / ppu,
                    (gp.y - pivot.y) / ppu
                ));
            }
            return result;
        }

        // ─────────────────────────────────────────────────────────────────
        //  Asset 路径
        // ─────────────────────────────────────────────────────────────────

        private static string GetTargetAssetPath(RCWBObject obj)
        {
            var scene = obj.gameObject.scene;
            if (string.IsNullOrEmpty(scene.path))
                return null;

            string sceneDir  = Path.GetDirectoryName(scene.path).Replace('\\', '/');
            string sceneName = Path.GetFileNameWithoutExtension(scene.path);
            string goName    = SanitizeFileName(obj.gameObject.name);

            var settings = PolygonManagerSettings.Instance;
            if (settings != null && !string.IsNullOrEmpty(settings.defaultProfileFolder))
            {
                // 方案 D：用户自定义根目录
                string root = settings.defaultProfileFolder.TrimEnd('/');
                return $"{root}/{sceneName}/{goName}_ContourProfile.asset";
            }
            else
            {
                // 方案 A：紧邻场景目录
                return $"{sceneDir}/{sceneName}_RCWBProfiles/{goName}_ContourProfile.asset";
            }
        }

        private static string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');
            return name;
        }

        private static void CreateFolderRecursive(string folderPath)
        {
            string[] parts   = folderPath.Split('/');
            string   current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }

        // ─────────────────────────────────────────────────────────────────
        //  源 rect（与 SpritePhysicsShapeGenerator 完全一致）
        // ─────────────────────────────────────────────────────────────────

        private static Rect GetSourceRect(Sprite sprite, TextureImporter importer)
        {
            if (importer.spriteImportMode == SpriteImportMode.Multiple)
            {
                foreach (var sheet in importer.spritesheet)
                {
                    if (sheet.name == sprite.name)
                        return sheet.rect;
                }
            }
            var srcTex = AssetDatabase.LoadAssetAtPath<Texture2D>(AssetDatabase.GetAssetPath(sprite));
            return new Rect(0, 0, srcTex.width, srcTex.height);
        }

        // ─────────────────────────────────────────────────────────────────
        //  遮罩、轮廓追踪、简化（与 SpritePhysicsShapeGenerator 完全一致）
        // ─────────────────────────────────────────────────────────────────

        private bool[,] BuildMask(Color[] pixels, int w, int h)
        {
            bool[,] mask = new bool[w, h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    mask[x, y] = pixels[y * w + x].a > alphaThreshold;
            return mask;
        }

        private enum Dir { Right = 0, Up = 1, Left = 2, Down = 3 }

        private static readonly int[] dx = { 1, 0, -1, 0 };
        private static readonly int[] dy = { 0, 1, 0, -1 };

        private List<List<Vector2Int>> TraceAllContours(bool[,] mask, int w, int h)
        {
            bool[,,] edgeVisited = new bool[w + 1, h + 1, 4];
            var allLoops = new List<List<Vector2Int>>();

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

        private bool IsBoundaryEdge(bool[,] mask, int w, int h, int gx, int gy, Dir dir)
        {
            GetEdgeNeighborPixels(gx, gy, dir, out int lx, out int ly, out int rx, out int ry);
            bool leftSolid  = InBounds(lx, ly, w, h) && mask[lx, ly];
            bool rightSolid = InBounds(rx, ry, w, h) && mask[rx, ry];
            return leftSolid && !rightSolid;
        }

        private static void GetEdgeNeighborPixels(
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

        private List<Vector2Int> TraceSingleContour(
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

                int ngx = gx + dx[(int)dir];
                int ngy = gy + dy[(int)dir];
                dir = ChooseNextDirection(mask, w, h, ngx, ngy, dir);
                gx  = ngx;
                gy  = ngy;

                if (gx == startGx && gy == startGy && dir == startDir) break;
            }

            return loop;
        }

        private Dir ChooseNextDirection(bool[,] mask, int w, int h, int gx, int gy, Dir currentDir)
        {
            Dir rightTurn = (Dir)(((int)currentDir + 3) % 4);
            Dir straight  = currentDir;
            Dir leftTurn  = (Dir)(((int)currentDir + 1) % 4);
            Dir uTurn     = (Dir)(((int)currentDir + 2) % 4);

            if (IsBoundaryEdge(mask, w, h, gx, gy, rightTurn)) return rightTurn;
            if (IsBoundaryEdge(mask, w, h, gx, gy, straight))  return straight;
            if (IsBoundaryEdge(mask, w, h, gx, gy, leftTurn))  return leftTurn;
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

                bool collinearX = prev.x == curr.x && curr.x == next.x;
                bool collinearY = prev.y == curr.y && curr.y == next.y;

                if (!collinearX && !collinearY)
                    result.Add(curr);
            }
            return result;
        }
    }
}

#endif
