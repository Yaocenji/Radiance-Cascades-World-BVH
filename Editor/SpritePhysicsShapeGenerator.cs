using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditor.U2D.Sprites;

namespace RadianceCascadesWorldBVH.Editor
{
    public class SpritePhysicsShapeGenerator : EditorWindow
    {
        private Sprite targetSprite;
        private float luminanceThreshold = 0.5f;
        private bool useAlphaOnly = true;
        private float alphaThreshold = 0.01f;

        [MenuItem("Tools/RCWB/Sprite Physics Shape Generator")]
        public static void ShowWindow()
        {
            GetWindow<SpritePhysicsShapeGenerator>("Physics Shape Generator");
        }

        private void OnGUI()
        {
            GUILayout.Label("Sprite Custom Physics Shape 生成器", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            targetSprite = (Sprite)EditorGUILayout.ObjectField("目标 Sprite", targetSprite, typeof(Sprite), false);

            EditorGUILayout.Space();
            useAlphaOnly = EditorGUILayout.Toggle("仅使用 Alpha 通道", useAlphaOnly);

            if (useAlphaOnly)
                alphaThreshold = EditorGUILayout.Slider("Alpha 阈值", alphaThreshold, 0f, 1f);
            else
                luminanceThreshold = EditorGUILayout.Slider("明度阈值", luminanceThreshold, 0f, 1f);

            EditorGUILayout.Space();

            EditorGUI.BeginDisabledGroup(targetSprite == null);
            if (GUILayout.Button("生成 Custom Physics Shape", GUILayout.Height(30)))
            {
                GeneratePhysicsShape(targetSprite);
            }
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "工作流程：\n" +
                "1. 从源纹理（非Atlas）读取 Sprite 像素\n" +
                "2. 按阈值二值化\n" +
                "3. 追踪像素边界轮廓（沿像素格点走）\n" +
                "4. 去除共线冗余顶点\n" +
                "5. 写入 Custom Physics Shape",
                MessageType.Info);
        }

        /// <summary>
        /// 从源资产获取 Sprite 在原始纹理中的 rect。
        /// 当 Sprite 被加入 SpriteAtlas 后，sprite.rect 可能指向 Atlas 纹理，
        /// 因此必须从 TextureImporter 的 spritesheet 中获取原始 rect。
        /// </summary>
        private Rect GetSourceRect(Sprite sprite, TextureImporter importer)
        {
            if (importer.spriteImportMode == SpriteImportMode.Multiple)
            {
                foreach (var sheet in importer.spritesheet)
                {
                    if (sheet.name == sprite.name)
                        return sheet.rect;
                }
            }

            // Single sprite 模式或未找到匹配：整张纹理
            var srcTex = AssetDatabase.LoadAssetAtPath<Texture2D>(AssetDatabase.GetAssetPath(sprite));
            return new Rect(0, 0, srcTex.width, srcTex.height);
        }

        private void GeneratePhysicsShape(Sprite sprite)
        {
            string path = AssetDatabase.GetAssetPath(sprite);
            var importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer == null)
            {
                Debug.LogError("无法获取 TextureImporter");
                return;
            }

            // 始终从源纹理读取，而非 sprite.texture（后者在Atlas打包后指向Atlas纹理）
            var sourceTex = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (sourceTex == null)
            {
                Debug.LogError($"无法加载源纹理：{path}");
                return;
            }
            if (!sourceTex.isReadable)
            {
                Debug.LogError($"纹理 '{sourceTex.name}' 不可读。请在 Import Settings 中勾选 Read/Write。");
                return;
            }

            Rect sourceRect = GetSourceRect(sprite, importer);
            int x0 = Mathf.FloorToInt(sourceRect.x);
            int y0 = Mathf.FloorToInt(sourceRect.y);
            int w = Mathf.FloorToInt(sourceRect.width);
            int h = Mathf.FloorToInt(sourceRect.height);

            Color[] pixels = sourceTex.GetPixels(x0, y0, w, h);
            bool[,] mask = BuildMask(pixels, w, h);

            List<List<Vector2Int>> rawLoops = TraceAllContours(mask, w, h);

            if (rawLoops.Count == 0)
            {
                Debug.LogWarning("未检测到任何轮廓，请检查阈值设置。");
                return;
            }

            List<List<Vector2>> simplifiedLoops = new List<List<Vector2>>();
            foreach (var loop in rawLoops)
            {
                var simplified = RemoveCollinearPoints(loop);
                if (simplified.Count >= 3)
                {
                    var floatLoop = ConvertToSpriteLocalSpace(simplified, sourceRect);
                    simplifiedLoops.Add(floatLoop);
                }
            }

            if (simplifiedLoops.Count == 0)
            {
                Debug.LogWarning("简化后无有效轮廓。");
                return;
            }

            ApplyPhysicsShape(sprite, importer, simplifiedLoops);

            int totalVerts = 0;
            foreach (var l in simplifiedLoops) totalVerts += l.Count;
            Debug.Log($"成功生成 Custom Physics Shape：{simplifiedLoops.Count} 个轮廓，共 {totalVerts} 个顶点。");
        }

        /// <summary>
        /// 根据阈值构建二值化遮罩
        /// </summary>
        private bool[,] BuildMask(Color[] pixels, int w, int h)
        {
            bool[,] mask = new bool[w, h];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    Color c = pixels[y * w + x];
                    if (useAlphaOnly)
                        mask[x, y] = c.a > alphaThreshold;
                    else
                        mask[x, y] = c.a > 0.001f && (0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b) > luminanceThreshold;
                }
            }
            return mask;
        }

        // ─────────────────────────────────────────────────────
        //  轮廓追踪：在像素格点上沿着 solid/empty 边界行走
        //  格点坐标 (gx, gy) 对应像素 (gx-1,gy-1), (gx,gy-1), (gx-1,gy), (gx,gy) 四个角
        //  格点范围 [0..w] x [0..h]
        // ─────────────────────────────────────────────────────

        private enum Dir { Right = 0, Up = 1, Left = 2, Down = 3 }

        private static readonly int[] dx = { 1, 0, -1, 0 };
        private static readonly int[] dy = { 0, 1, 0, -1 };

        /// <summary>
        /// 追踪所有独立轮廓
        /// </summary>
        private List<List<Vector2Int>> TraceAllContours(bool[,] mask, int w, int h)
        {
            // edgeVisited[gx, gy, dir] 标记从格点(gx,gy)沿dir方向的边是否已访问
            bool[,,] edgeVisited = new bool[w + 1, h + 1, 4];
            var allLoops = new List<List<Vector2Int>>();

            // 扫描所有格点边，找到未访问的边界边作为起点
            for (int gy = 0; gy <= h; gy++)
            {
                for (int gx = 0; gx <= w; gx++)
                {
                    for (int d = 0; d < 4; d++)
                    {
                        if (edgeVisited[gx, gy, d]) continue;
                        if (!IsBoundaryEdge(mask, w, h, gx, gy, (Dir)d)) continue;

                        var loop = TraceSingleContour(mask, w, h, edgeVisited, gx, gy, (Dir)d);
                        if (loop != null && loop.Count >= 4)
                            allLoops.Add(loop);
                    }
                }
            }

            return allLoops;
        }

        /// <summary>
        /// 判断从格点(gx,gy)沿dir方向的边是否是 solid/empty 边界，
        /// 且满足"左侧solid、右侧empty"的约定
        /// </summary>
        private bool IsBoundaryEdge(bool[,] mask, int w, int h, int gx, int gy, Dir dir)
        {
            GetEdgeNeighborPixels(gx, gy, dir, out int lx, out int ly, out int rx, out int ry);
            bool leftSolid = InBounds(lx, ly, w, h) && mask[lx, ly];
            bool rightSolid = InBounds(rx, ry, w, h) && mask[rx, ry];
            return leftSolid && !rightSolid;
        }

        /// <summary>
        /// 获取边两侧的像素坐标。
        /// 约定：沿行走方向，左侧为 solid 侧（保证逆时针绕行solid区域外边界）
        /// </summary>
        private void GetEdgeNeighborPixels(int gx, int gy, Dir dir, out int lx, out int ly, out int rx, out int ry)
        {
            // 沿dir行走时，左侧像素和右侧像素
            switch (dir)
            {
                case Dir.Right: // 从(gx,gy)到(gx+1,gy)，左侧=上方像素(gx, gy)，右侧=下方像素(gx, gy-1)
                    lx = gx; ly = gy; rx = gx; ry = gy - 1; break;
                case Dir.Up:    // 从(gx,gy)到(gx,gy+1)，左侧=左方像素(gx-1, gy)，右侧=右方像素(gx, gy)
                    lx = gx - 1; ly = gy; rx = gx; ry = gy; break;
                case Dir.Left:  // 从(gx,gy)到(gx-1,gy)，左侧=下方像素(gx-1, gy-1)，右侧=上方像素(gx-1, gy)
                    lx = gx - 1; ly = gy - 1; rx = gx - 1; ry = gy; break;
                case Dir.Down:  // 从(gx,gy)到(gx,gy-1)，左侧=右方像素(gx, gy-1)，右侧=左方像素(gx-1, gy-1)
                    lx = gx; ly = gy - 1; rx = gx - 1; ry = gy - 1; break;
                default:
                    lx = ly = rx = ry = 0; break;
            }
        }

        /// <summary>
        /// 从指定起始边开始追踪一条完整轮廓。
        /// solid 始终在行走方向的左侧，优先右转以紧贴 solid 外边界。
        /// </summary>
        private List<Vector2Int> TraceSingleContour(bool[,] mask, int w, int h,
            bool[,,] edgeVisited, int startGx, int startGy, Dir startDir)
        {
            var loop = new List<Vector2Int>();
            int gx = startGx, gy = startGy;
            Dir dir = startDir;

            int maxSteps = (w + 1) * (h + 1) * 4 + 1;
            for (int step = 0; step < maxSteps; step++)
            {
                if (edgeVisited[gx, gy, (int)dir])
                {
                    if (gx == startGx && gy == startGy && dir == startDir && loop.Count > 0)
                        break;
                    if (loop.Count > 0) break;
                    return null;
                }

                edgeVisited[gx, gy, (int)dir] = true;
                loop.Add(new Vector2Int(gx, gy));

                int ngx = gx + dx[(int)dir];
                int ngy = gy + dy[(int)dir];

                // 到达下一个格点后，决定转向
                dir = ChooseNextDirection(mask, w, h, ngx, ngy, dir);
                gx = ngx;
                gy = ngy;

                if (gx == startGx && gy == startGy && dir == startDir)
                    break;
            }

            return loop;
        }

        /// <summary>
        /// 在格点(gx,gy)处，根据当前行走方向选择下一步方向。
        /// 优先级：右转 > 直行 > 左转 > 掉头（紧贴 solid 外边界）
        /// </summary>
        private Dir ChooseNextDirection(bool[,] mask, int w, int h, int gx, int gy, Dir currentDir)
        {
            Dir rightTurn = (Dir)(((int)currentDir + 3) % 4);
            Dir straight = currentDir;
            Dir leftTurn = (Dir)(((int)currentDir + 1) % 4);
            Dir uTurn = (Dir)(((int)currentDir + 2) % 4);

            if (IsBoundaryEdge(mask, w, h, gx, gy, rightTurn)) return rightTurn;
            if (IsBoundaryEdge(mask, w, h, gx, gy, straight)) return straight;
            if (IsBoundaryEdge(mask, w, h, gx, gy, leftTurn)) return leftTurn;
            return uTurn;
        }

        private static bool InBounds(int x, int y, int w, int h)
        {
            return x >= 0 && x < w && y >= 0 && y < h;
        }

        // ─────────────────────────────────────────────────────
        //  轮廓简化：去除共线冗余点
        // ─────────────────────────────────────────────────────

        /// <summary>
        /// 去除共线的冗余顶点。因为轮廓沿像素格点走，只有水平和垂直线段，
        /// 所以共线判断就是检查三个连续点是否在同一水平线或垂直线上。
        /// </summary>
        private List<Vector2Int> RemoveCollinearPoints(List<Vector2Int> loop)
        {
            if (loop.Count < 3) return loop;

            var result = new List<Vector2Int>();
            int n = loop.Count;

            for (int i = 0; i < n; i++)
            {
                Vector2Int prev = loop[(i - 1 + n) % n];
                Vector2Int curr = loop[i];
                Vector2Int next = loop[(i + 1) % n];

                bool collinearX = (prev.x == curr.x && curr.x == next.x);
                bool collinearY = (prev.y == curr.y && curr.y == next.y);

                if (!collinearX && !collinearY)
                    result.Add(curr);
            }

            return result;
        }

        // ─────────────────────────────────────────────────────
        //  坐标转换与输出
        // ─────────────────────────────────────────────────────

        /// <summary>
        /// 将格点坐标转换为 ISpritePhysicsOutlineDataProvider 所需的坐标空间。
        /// 该坐标系以源 rect 中心为原点，单位为像素。
        /// 格点 (0,0) 对应 rect 的左下角。
        /// </summary>
        private List<Vector2> ConvertToSpriteLocalSpace(List<Vector2Int> gridPoints, Rect sourceRect)
        {
            float halfW = sourceRect.width * 0.5f;
            float halfH = sourceRect.height * 0.5f;

            var result = new List<Vector2>(gridPoints.Count);
            foreach (var gp in gridPoints)
            {
                result.Add(new Vector2(gp.x - halfW, gp.y - halfH));
            }
            return result;
        }

        /// <summary>
        /// 通过 ISpritePhysicsOutlineDataProvider 将轮廓写入 Custom Physics Shape
        /// </summary>
        private void ApplyPhysicsShape(Sprite sprite, TextureImporter importer, List<List<Vector2>> outlines)
        {
            var factory = new SpriteDataProviderFactories();
            factory.Init();
            var dataProvider = factory.GetSpriteEditorDataProviderFromObject(importer);
            dataProvider.InitSpriteEditorDataProvider();

            var physicsProvider = dataProvider.GetDataProvider<ISpritePhysicsOutlineDataProvider>();
            if (physicsProvider == null)
            {
                Debug.LogError("无法获取 ISpritePhysicsOutlineDataProvider");
                return;
            }

            var spriteRects = dataProvider.GetSpriteRects();
            GUID targetGuid = default;
            bool found = false;

            // 优先按 sprite name 匹配（最可靠，不受 Atlas 影响）
            foreach (var sr in spriteRects)
            {
                if (sr.name == sprite.name)
                {
                    targetGuid = sr.spriteID;
                    found = true;
                    break;
                }
            }

            // 回退：Single 模式下只有一个 SpriteRect
            if (!found && spriteRects.Length == 1)
            {
                targetGuid = spriteRects[0].spriteID;
                found = true;
            }

            if (!found)
            {
                Debug.LogError($"在 SpriteEditorDataProvider 中未找到名为 '{sprite.name}' 的 SpriteRect");
                return;
            }

            var outlineArrays = new List<Vector2[]>();
            foreach (var loop in outlines)
                outlineArrays.Add(loop.ToArray());

            physicsProvider.SetOutlines(targetGuid, outlineArrays);
            dataProvider.Apply();

            importer.SaveAndReimport();
        }
    }
}
