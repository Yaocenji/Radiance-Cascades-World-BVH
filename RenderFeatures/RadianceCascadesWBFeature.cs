using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace RadianceCascadesWorldBVH
{

    [Serializable]
    public class RcwbSettings
    {
        [Header("基础渲染设置")]
        public float renderScale;
        public int cascadeCount;
        public float rayRange;
        public float bounceIntensity;

        [Header("天空与太阳")]
        public Color skyColor;
        public float skyIntensity;
        public Color sunColor;
        public float sunAngle;
        public float sunIntensity;
        public float sunHardness;

        [Header("模糊设置")]
        public int blurIterations = 4;      // 控制降采样的深度，4 是个非常好的默认值
        public float blurRadius = 1.0f;     // 散布半径

        [Header("全局光高度")]
        public float giHeight = 1.0f;

        [Header("Shading Options")]
        [Tooltip("Enable translucent-object branch in shader space (keyword: ENABLE_TRANSLUCENT_OBJECTS).")]
        public bool enableTranslucentObjects = false;

        public RcwbSettings(
            float renderScale,
            int cascadeCount,
            float rayRange,
            float bounceIntensity,
            Color skyColor,
            float skyIntensity,
            Color sunColor,
            float sunAngle,
            float sunIntensity,
            float sunHardness,
            float giHeight)
        {
            this.renderScale = renderScale;
            this.cascadeCount = cascadeCount;
            this.rayRange = rayRange;
            this.bounceIntensity = bounceIntensity;
            this.skyColor = skyColor;
            this.skyIntensity = skyIntensity;
            this.sunColor = sunColor;
            this.sunAngle = sunAngle;
            this.sunIntensity = sunIntensity;
            this.sunHardness = sunHardness;
            this.giHeight = giHeight;
        }
    }
    
    public class RadianceCascadesWBFeature : ScriptableRendererFeature
    {
        private const string TranslucentObjectsKeyword = "ENABLE_TRANSLUCENT_OBJECTS";
        public ComputeShader rcShader;
        public RcwbSettings settings;
        
        class RcwbRenderPass : ScriptableRenderPass
        {
            public ComputeShader rcShader;
            public RcwbSettings settings;
            
            // 摄像机的引用
            private Camera m_Camera;

            // 历史帧 RT 引用（由外部赋值）
            public RTHandle historyRT;
            
            // RT的引用
            private RTHandle m_Rcwb_Handle_0;
            private RTHandle m_Rcwb_Handle_1;
            //private RTHandle m_Rcwb_Handle_StencilFullResolution;
            private RTHandle m_Rcwb_Direction;
            private RTHandle m_Rcwb_Direction_Blur;
            private RTHandle[] m_KawasePyramid;
            
            // 上一帧的 VP / InvVP 矩阵（用于 reprojection）
            private Matrix4x4 m_PrevViewProjMatrix    = Matrix4x4.identity;
            private Matrix4x4 m_PrevViewProjMatrixInv = Matrix4x4.identity;

            // 当前的目标分辨率与RC分辨率
            private int fullWidth;
            private int fullHeight;
            private int width;
            private int height;
            private int rcWidth;
            private int rcHeight;

            public RcwbRenderPass(RcwbSettings settings, ComputeShader rcShader)
            {
                this.settings = settings;
                this.rcShader = rcShader;
            }
            
            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
            {
                m_Camera = renderingData.cameraData.camera;
                var cameraTargetDescriptor = renderingData.cameraData.cameraTargetDescriptor;
                
                fullWidth = cameraTargetDescriptor.width;
                fullHeight = cameraTargetDescriptor.height;

                // 计算目标分辨率（支持动态画质调整）
                width = (int)(cameraTargetDescriptor.width * settings.renderScale);
                height = (int)(cameraTargetDescriptor.height * settings.renderScale);
                
                // 计算rc分辨率
                int blockSize = (int)Mathf.Pow(2, settings.cascadeCount);
                rcWidth = Mathf.CeilToInt((float)width / blockSize) * blockSize;
                rcHeight = Mathf.CeilToInt((float)height / blockSize) * blockSize;
                
                // 配置 Radiance Buffer 描述符
                var radianceDesc = new RenderTextureDescriptor(rcWidth, rcHeight, RenderTextureFormat.ARGBHalf, 0);
                radianceDesc.msaaSamples = 1;
                radianceDesc.sRGB = false;
                radianceDesc.useMipMap = false;
                radianceDesc.enableRandomWrite = true;

                // 配置 Radiance Full Resolution 描述符
                var radianceFullResolutionDesc = new RenderTextureDescriptor(fullWidth, fullHeight, RenderTextureFormat.RHalf, 0);
                radianceFullResolutionDesc.msaaSamples = 1;
                radianceFullResolutionDesc.sRGB = false;
                radianceFullResolutionDesc.useMipMap = false;
                radianceFullResolutionDesc.enableRandomWrite = true;

                // 配置 Radiance Direction 描述符
                var radianceDirectionDesc = new RenderTextureDescriptor(rcWidth, rcHeight, RenderTextureFormat.RGHalf, 0);
                radianceDirectionDesc.msaaSamples = 1;
                radianceDirectionDesc.sRGB = false;
                radianceDirectionDesc.useMipMap = false;
                radianceDirectionDesc.enableRandomWrite = true;
                
                // 动态地分配显存
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_0, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_0");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_1, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_1");
                //RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_StencilFullResolution, radianceFullResolutionDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_FullResolution");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Direction, radianceDirectionDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_Direction");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Direction_Blur, radianceDirectionDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_Direction_Blur");

                // 特殊的：分配降采样纹理的金字塔缓存
                int maxIterations = settings.blurIterations;
                if (m_KawasePyramid == null || m_KawasePyramid.Length != maxIterations)
                {
                    m_KawasePyramid = new RTHandle[maxIterations];
                }
                int mipWidth = rcWidth / 2;
                int mipHeight = rcHeight / 2;
                var mipDesc = radianceDesc; // 继承之前配置好的 Descriptor
                for (int i = 0; i < maxIterations; i++)
                {
                    mipDesc.width = Mathf.Max(1, mipWidth);
                    mipDesc.height = Mathf.Max(1, mipHeight);
                    
                    RenderingUtils.ReAllocateIfNeeded(
                        ref m_KawasePyramid[i], 
                        mipDesc, 
                        FilterMode.Bilinear, 
                        TextureWrapMode.Clamp, 
                        name: $"_RCWB_KawaseMip_{i}"
                    );

                    mipWidth /= 2;
                    mipHeight /= 2;
                }
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                // 写个名字
                CommandBuffer cmd = CommandBufferPool.Get("Radiance Cascades");
                if (settings != null && settings.enableTranslucentObjects)
                {
                    cmd.EnableShaderKeyword(TranslucentObjectsKeyword);
                }
                else
                {
                    cmd.DisableShaderKeyword(TranslucentObjectsKeyword);
                }
                
                cmd.BeginSample("Radiance Cascades WB");
                
                int rcMainKernelHandle = rcShader.FindKernel("RadianceCascadeMain");
                
                // 摄像机的VP矩阵及其逆矩阵
                Matrix4x4 view = m_Camera.worldToCameraMatrix;
                Matrix4x4 proj = GL.GetGPUProjectionMatrix(m_Camera.projectionMatrix, true);
                Matrix4x4 viewProjMatrix = proj * view;
                Matrix4x4 viewProjMatrixInv = Matrix4x4.Inverse(viewProjMatrix);
                cmd.SetComputeMatrixParam(rcShader, "MatrixVP", viewProjMatrix);
                cmd.SetComputeMatrixParam(rcShader, "MatrixInvVP", viewProjMatrixInv);
                cmd.SetComputeMatrixParam(rcShader, "MatrixVP_Prev", m_PrevViewProjMatrix);

                // 显式绑定历史帧纹理到 compute kernel
                cmd.SetComputeTextureParam(rcShader, rcMainKernelHandle, "_RCWB_HistoryColor", historyRT);
                
                cmd.SetComputeVectorParam(rcShader, "_CameraResolution_Full", new Vector2(fullWidth, fullHeight));
                cmd.SetComputeVectorParam(rcShader, "_CameraResolution_Resized", new Vector2(width, height));

                // RC分辨率
                cmd.SetComputeVectorParam(rcShader, "_RCWB_CascadeResolution", new Vector2(rcWidth, rcHeight));
                
                // RC最大层级
                cmd.SetComputeIntParam(rcShader, "_RCWB_CascadeCount", settings.cascadeCount);
                
                // 射线范围
                cmd.SetComputeFloatParam(rcShader, "_RCWB_RayRange_WS", settings.rayRange);
                cmd.SetComputeVectorParam(rcShader, "_RCWB_SkyColor", settings.skyColor);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SkyIntensity", settings.skyIntensity);
                cmd.SetComputeVectorParam(rcShader, "_RCWB_SunColor", settings.sunColor);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SunAngle", settings.sunAngle);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SunIntensity", settings.sunIntensity);
                cmd.SetComputeFloatParam(rcShader, "_RCWB_SunHardness", settings.sunHardness);

                for (int i = 0; i < settings.cascadeCount; i++)
                {
                    // RC的当前层级
                    cmd.SetComputeIntParam(rcShader, "_RCWB_CascadeLevel", settings.cascadeCount - i - 1);
                    // RC的 merge 纹理
                    cmd.SetComputeTextureParam(rcShader, rcMainKernelHandle, "_RCWB_LastResult", (i & 1) == 1 ?  m_Rcwb_Handle_0 : m_Rcwb_Handle_1);
                    // RC的目标纹理
                    cmd.SetComputeTextureParam(rcShader, rcMainKernelHandle, "Result", (i & 1) == 1 ?  m_Rcwb_Handle_1 : m_Rcwb_Handle_0);
                    cmd.DispatchCompute(rcShader, rcMainKernelHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);
                }
                
                cmd.EndSample("Radiance Cascades WB");
                cmd.BeginSample("Calculate Direction");

                int directionKernalHandle = rcShader.FindKernel("DirectionMain");
                cmd.SetComputeTextureParam(rcShader, directionKernalHandle, "Direction", m_Rcwb_Direction);
                cmd.SetComputeTextureParam(rcShader, directionKernalHandle, "_RCWB_OneDirResult", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1);
                cmd.SetComputeTextureParam(rcShader, directionKernalHandle, "_RCWB_FourDirResult", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0);
                cmd.DispatchCompute(rcShader, directionKernalHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);

                cmd.EndSample("Calculate Direction");
                // Dual Kawase Blur 阶段
                cmd.BeginSample("Invasion");

                int invasionLightKernalHandle = rcShader.FindKernel("InvasionLightMain");
                cmd.SetComputeTextureParam(rcShader, invasionLightKernalHandle, "_RCWB_LightResult_NonInvasion", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1);
                cmd.SetComputeTextureParam(rcShader, invasionLightKernalHandle, "_RCWB_LightResult_Invasion", ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0);
                cmd.DispatchCompute(rcShader, invasionLightKernalHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);

                cmd.CopyTexture(((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0, 
                                ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1);

                int invasionDirectionKernalHandle = rcShader.FindKernel("InvasionDirectionMain");
                cmd.SetComputeTextureParam(rcShader, invasionDirectionKernalHandle, "_RCWB_Direction_NonInvasion", m_Rcwb_Direction);
                cmd.SetComputeTextureParam(rcShader, invasionDirectionKernalHandle, "_RCWB_Direction_Invasion", m_Rcwb_Direction_Blur);
                cmd.DispatchCompute(rcShader, invasionDirectionKernalHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);

                cmd.CopyTexture(m_Rcwb_Direction_Blur, m_Rcwb_Direction);
                
                cmd.EndSample("Invasion");
                cmd.BeginSample("Dual Kawase Blur");
                
                ExecuteDualKawaseBlur(cmd, ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1, ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0, rcWidth, rcHeight, settings.blurIterations, settings.blurRadius);
                ExecuteDualKawaseBlur(cmd, m_Rcwb_Direction, m_Rcwb_Direction_Blur, rcWidth, rcHeight, settings.blurIterations, settings.blurRadius);
                cmd.EndSample("Dual Kawase Blur");
                cmd.BeginSample("After RCWB");

                // 获取颜色结果
                // 奇数次是handle0，偶数次是handle1
                var lightResultHandle = ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1;
                var lightResultHandleBlur = ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_1 : m_Rcwb_Handle_0;
                
                //RTHandle cameraColorTargetHandle = renderingData.cameraData.renderer.cameraColorTargetHandle;
                //Blitter.BlitCameraTexture(cmd, m_Rcwb_Direction, cameraColorTargetHandle, 0f, true);

                // 将RC变量绑定到全局，让后续能访问到
                cmd.SetGlobalMatrix("MatrixVP", viewProjMatrix);
                cmd.SetGlobalMatrix("MatrixInvVP", viewProjMatrixInv);
                cmd.SetGlobalMatrix("MatrixVP_Prev", m_PrevViewProjMatrix);
                cmd.SetGlobalMatrix("MatrixInvVP_Prev", m_PrevViewProjMatrixInv);
                // 保存本帧矩阵供下一帧使用
                m_PrevViewProjMatrix    = viewProjMatrix;
                m_PrevViewProjMatrixInv = viewProjMatrixInv;
                cmd.SetGlobalVector("_CameraResolution_Full", new Vector2(fullWidth, fullHeight));
                cmd.SetGlobalVector("_CameraResolution_Resized", new Vector2(width, height));
                cmd.SetGlobalVector("_RCWB_CascadeResolution", new Vector2(rcWidth, rcHeight));
                cmd.SetGlobalInt("_RCWB_CascadeCount", settings.cascadeCount);
                cmd.SetGlobalFloat("_RCWB_RayRange_WS", settings.rayRange);
                // 包含全局光的高度
                cmd.SetGlobalFloat("_RCWB_GI_Height", settings.giHeight);

                // 最后的最后：将光源和方向的纹理绑定，让后续的渲染中能够采样到
                cmd.SetGlobalTexture("_RCWB_LightResult", lightResultHandle);
                cmd.SetGlobalTexture("_RCWB_LightResult_Blur", lightResultHandleBlur);
                cmd.SetGlobalTexture("_RCWB_DirectionResult", m_Rcwb_Direction);
                cmd.SetGlobalTexture("_RCWB_DirectionResult_Blur", m_Rcwb_Direction_Blur);
                
                cmd.EndSample("After RCWB");
                
                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
                
            }

            private void ExecuteDualKawaseBlur(CommandBuffer cmd, RTHandle srcHandle, RTHandle dstHandle, int width, int height, int iterations, float radius)
            {
                // 1. 健壮性检查：如果金字塔未分配，或者要求迭代次数 <= 0
                if (m_KawasePyramid == null || m_KawasePyramid.Length == 0 || iterations <= 0)
                {
                    // 直接将输入拷贝到输出（前提是它们不是同一个RT）
                    if (srcHandle != dstHandle)
                    {
                        Blitter.BlitCameraTexture(cmd, srcHandle, dstHandle, 0f, true);
                    }
                    return;
                }

                // 2. 防止传入的迭代次数超过实际分配的金字塔层数
                int actualIterations = Mathf.Min(iterations, m_KawasePyramid.Length);

                int downKernel = rcShader.FindKernel("KawaseDownsample");
                int upKernel = rcShader.FindKernel("KawaseUpsample");

                RTHandle currentSrc = srcHandle;
                int currentSrcWidth = width;
                int currentSrcHeight = height;

                // ==========================================
                // 阶段 1: 降采样 (Downsample)
                // ==========================================
                for (int i = 0; i < actualIterations; i++)
                {
                    RTHandle currentDst = m_KawasePyramid[i];
                    int dstWidth = currentDst.rt.width;
                    int dstHeight = currentDst.rt.height;

                    cmd.SetComputeVectorParam(rcShader, "_RCWB_InputResolution", new Vector2(currentSrcWidth, currentSrcHeight));
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_OutputResolution", new Vector2(dstWidth, dstHeight));
                    cmd.SetComputeTextureParam(rcShader, downKernel, "_RCWB_BlurInput", currentSrc);
                    cmd.SetComputeTextureParam(rcShader, downKernel, "_RCWB_BlurOutput", currentDst);
                    
                    cmd.DispatchCompute(rcShader, downKernel, (dstWidth + 7) / 8, (dstHeight + 7) / 8, 1);

                    currentSrc = currentDst;
                    currentSrcWidth = dstWidth;
                    currentSrcHeight = dstHeight;
                }

                // 传入自定义的模糊散布半径
                cmd.SetComputeFloatParam(rcShader, "_RCWB_BlurRadius", radius);

                // ==========================================
                // 阶段 2: 升采样 (Upsample) - 直到倒数第二次
                // ==========================================
                for (int i = actualIterations - 2; i >= 0; i--)
                {
                    RTHandle currentDst = m_KawasePyramid[i];
                    int dstWidth = currentDst.rt.width;
                    int dstHeight = currentDst.rt.height;

                    cmd.SetComputeVectorParam(rcShader, "_RCWB_InputResolution", new Vector2(currentSrcWidth, currentSrcHeight));
                    cmd.SetComputeVectorParam(rcShader, "_RCWB_OutputResolution", new Vector2(dstWidth, dstHeight));
                    cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurInput", currentSrc);
                    cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurOutput", currentDst);
                    
                    cmd.DispatchCompute(rcShader, upKernel, (dstWidth + 7) / 8, (dstHeight + 7) / 8, 1);

                    currentSrc = currentDst;
                    currentSrcWidth = dstWidth;
                    currentSrcHeight = dstHeight;
                }

                // ==========================================
                // 阶段 3: 最后一次升采样 (直接输出到 dstHandle)
                // ==========================================
                cmd.SetComputeVectorParam(rcShader, "_RCWB_InputResolution", new Vector2(currentSrcWidth, currentSrcHeight));
                cmd.SetComputeVectorParam(rcShader, "_RCWB_OutputResolution", new Vector2(width, height));
                cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurInput", currentSrc);
                cmd.SetComputeTextureParam(rcShader, upKernel, "_RCWB_BlurOutput", dstHandle);
                
                cmd.DispatchCompute(rcShader, upKernel, (width + 7) / 8, (height + 7) / 8, 1);
            }

            public override void OnCameraCleanup(CommandBuffer cmd)
            {
            }
        }

        // =====================================================================
        // HistoryCapturePass：渲染结束后将摄像机颜色 blit 到跨帧持久 RT
        // =====================================================================
        class HistoryCapturePass : ScriptableRenderPass
        {
            /// <summary>相对于屏幕分辨率的缩放系数，默认 0.5（半分辨率）</summary>
            public float historyScale = 0.5f;

            private RTHandle m_HistoryRT;
            public RTHandle HistoryRT => m_HistoryRT;

            public HistoryCapturePass()
            {
                // 捕获点必须在 post-processing 之前：此时 camera color 仍是线性 HDR，
                // 避免 tonemap 压缩让反馈闭环陷入非线性色阶分层。
                renderPassEvent = RenderPassEvent.AfterRenderingTransparents;
            }

            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
            {
                var srcDesc = renderingData.cameraData.cameraTargetDescriptor;
                int w = Mathf.Max(1, Mathf.RoundToInt(srcDesc.width  * historyScale));
                int h = Mathf.Max(1, Mathf.RoundToInt(srcDesc.height * historyScale));

                var desc = new RenderTextureDescriptor(w, h, RenderTextureFormat.ARGBHalf, 0)
                {
                    msaaSamples = 1,
                    sRGB        = false,
                    useMipMap   = false
                };

                RenderingUtils.ReAllocateIfNeeded(
                    ref m_HistoryRT, desc,
                    FilterMode.Bilinear, TextureWrapMode.Clamp,
                    name: "_HistoryCaptureRT");
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                if (m_HistoryRT == null) return;

                var cmd = CommandBufferPool.Get("History Capture");
                cmd.BeginSample("History Capture");

                RTHandle cameraColor = renderingData.cameraData.renderer.cameraColorTargetHandle;
                Blitter.BlitCameraTexture(cmd, cameraColor, m_HistoryRT);
                cmd.SetGlobalTexture("_RCWB_HistoryColor", m_HistoryRT);
                // history RT 的分辨率，供 shader 做 texel snap
                int w = m_HistoryRT.rt.width;
                int h = m_HistoryRT.rt.height;
                cmd.SetGlobalVector("_RCWB_HistoryColor_TexelSize", new Vector4(1f / w, 1f / h, w, h));

                cmd.EndSample("History Capture");
                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
            }

            public override void OnCameraCleanup(CommandBuffer cmd) { }

            public void Dispose()
            {
                m_HistoryRT?.Release();
                m_HistoryRT = null;
            }
        }

        // =====================================================================

        RcwbRenderPass m_ScriptablePass;

        [Header("历史帧捕获")]
        [Tooltip("历史帧 RT 相对屏幕分辨率的缩放系数")]
        [Range(0.1f, 1f)]
        public float historyScale = 0.5f;

        private HistoryCapturePass m_HistoryCapturePass;

        /// <inheritdoc/>
        public override void Create()
        {
            m_ScriptablePass = new RcwbRenderPass(settings, rcShader);
            m_ScriptablePass.renderPassEvent = RenderPassEvent.BeforeRenderingOpaques;

            m_HistoryCapturePass = new HistoryCapturePass();

            ApplyShaderKeywords();
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            // 判定当前渲染窗口是哪个
            ApplyShaderKeywords();
            var cameraType = renderingData.cameraData.cameraType;
            if (cameraType == CameraType.Preview || cameraType == CameraType.Reflection || cameraType == CameraType.SceneView)
            {
                return;
            }

            // 每帧同步可能在 Inspector 中调整的参数
            m_HistoryCapturePass.historyScale       = historyScale;

            // 把历史帧 RT 传给 RC pass，用于 compute kernel 显式绑定
            m_ScriptablePass.historyRT = m_HistoryCapturePass.HistoryRT;

            // 只有 game 窗口会应用 renderPass
            renderer.EnqueuePass(m_ScriptablePass);
            renderer.EnqueuePass(m_HistoryCapturePass);
        }

        protected override void Dispose(bool disposing)
        {
            m_HistoryCapturePass?.Dispose();
        }

        private void ApplyShaderKeywords()
        {
            if (settings != null && settings.enableTranslucentObjects)
            {
                Shader.EnableKeyword(TranslucentObjectsKeyword);
            }
            else
            {
                Shader.DisableKeyword(TranslucentObjectsKeyword);
            }
        }
    }
        
}


