
using UnityEngine;
using UnityEngine.Rendering;

#if UNITY_EDITOR
using UnityEditor;

namespace RadianceCascadesWorldBVH
{
    [InitializeOnLoad]
    internal static class RCWBScenePreviewKeywordController
    {
        private const string ScenePreviewKeyword = "RCWB_EDITOR_SCENE_PREVIEW";

        static RCWBScenePreviewKeywordController()
        {
            RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
            RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeAssemblyReload;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        private static void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            if (camera != null && camera.cameraType == CameraType.SceneView)
            {
                Shader.EnableKeyword(ScenePreviewKeyword);
            }
            else
            {
                Shader.DisableKeyword(ScenePreviewKeyword);
            }
        }

        private static void OnEndCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            if (camera != null && camera.cameraType == CameraType.SceneView)
            {
                Shader.DisableKeyword(ScenePreviewKeyword);
            }
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingEditMode ||
                state == PlayModeStateChange.EnteredPlayMode ||
                state == PlayModeStateChange.ExitingPlayMode ||
                state == PlayModeStateChange.EnteredEditMode)
            {
                Shader.DisableKeyword(ScenePreviewKeyword);
            }
        }

        private static void OnBeforeAssemblyReload()
        {
            RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
            AssemblyReloadEvents.beforeAssemblyReload -= OnBeforeAssemblyReload;
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            Shader.DisableKeyword(ScenePreviewKeyword);
        }
    }
}

#endif