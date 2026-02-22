using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace RadianceCascadesWorldBVH
{

    [Serializable]
    public class RcwbSettings
    {
        public float renderScale;
        public int cascadeCount;
        public float rayRange;
        public float bounceIntensity;
        public Color skyColor;
        public float skyIntensity;
        public Color sunColor;
        public float sunAngle;
        public float sunIntensity;
        public float sunHardness;

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
            float sunHardness)
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
        }
    }
    
    public class RadianceCascadesWBFeature : ScriptableRendererFeature
    {
        public ComputeShader rcShader;
        public RcwbSettings settings;
        
        class RcwbRenderPass : ScriptableRenderPass
        {
            public ComputeShader rcShader;
            public RcwbSettings settings;
            
            // 摄像机的引用
            private Camera m_Camera;
            
            // RT的引用
            private RTHandle m_Rcwb_Handle_0;
            private RTHandle m_Rcwb_Handle_1;
            
            // 当前的目标分辨率与RC分辨率
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
                
                // 动态地分配显存
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_0, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_0");
                RenderingUtils.ReAllocateIfNeeded(ref m_Rcwb_Handle_1, radianceDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_RCWB_1");
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                // 写个名字
                CommandBuffer cmd = CommandBufferPool.Get("Radiance Cascades");
                
                cmd.BeginSample("Radiance Cascades WB");
                
                int kernelHandle = rcShader.FindKernel("CSMain");
                
                // 摄像机的VP矩阵及其逆矩阵
                Matrix4x4 view = m_Camera.worldToCameraMatrix;
                Matrix4x4 proj = GL.GetGPUProjectionMatrix(m_Camera.projectionMatrix, true);
                Matrix4x4 viewProjMatrix = proj * view;
                Matrix4x4 viewProjMatrixInv = Matrix4x4.Inverse(viewProjMatrix);
                cmd.SetComputeMatrixParam(rcShader, "MatrixVP", viewProjMatrix);
                cmd.SetComputeMatrixParam(rcShader, "MatrixInvVP", viewProjMatrixInv);
                
                // RC分辨率
                cmd.SetComputeVectorParam(rcShader, "_RCWB_CascadeResolution", new Vector2(rcWidth, rcHeight));
                
                // RC最大层级
                cmd.SetComputeIntParam(rcShader, "_RCWB_CascadeCount", settings.cascadeCount);
                
                // 
                cmd.SetComputeFloatParam(rcShader, "_RCWB_RayRange_WS", settings.rayRange);

                for (int i = 0; i < settings.cascadeCount; i++)
                {
                    // RC的当前层级
                    cmd.SetComputeIntParam(rcShader, "_RCWB_CascadeLevel", settings.cascadeCount - i - 1);
                    // RC的 merge 纹理
                    cmd.SetComputeTextureParam(rcShader, kernelHandle, "_RCWB_LastResult", (i & 1) == 1 ?  m_Rcwb_Handle_0 : m_Rcwb_Handle_1);
                    // RC的目标纹理
                    cmd.SetComputeTextureParam(rcShader, kernelHandle, "Result", (i & 1) == 1 ?  m_Rcwb_Handle_1 : m_Rcwb_Handle_0);
                    cmd.DispatchCompute(rcShader, kernelHandle, (rcWidth + 7) / 8, (rcHeight + 7) / 8, 1);
                }
                
                cmd.EndSample("Radiance Cascades WB");
                cmd.BeginSample("After RCWB");
                
                RTHandle cameraColorTargetHandle = renderingData.cameraData.renderer.cameraColorTargetHandle;
                // 奇数次是handle0，偶数次是handle1
                var ansHandle = ((settings.cascadeCount & 1) == 1) ? m_Rcwb_Handle_0 : m_Rcwb_Handle_1;
                Blitter.BlitCameraTexture(cmd, ansHandle, cameraColorTargetHandle);
                
                cmd.EndSample("After RCWB");
                
                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
                
            }

            public override void OnCameraCleanup(CommandBuffer cmd)
            {
            }
        }

        RcwbRenderPass m_ScriptablePass;

        /// <inheritdoc/>
        public override void Create()
        {
            m_ScriptablePass = new RcwbRenderPass(settings, rcShader);

            m_ScriptablePass.renderPassEvent = RenderPassEvent.BeforeRenderingOpaques;
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            // 判定当前渲染窗口是哪个
            var cameraType = renderingData.cameraData.cameraType;
            if (cameraType == CameraType.Preview || cameraType == CameraType.Reflection || cameraType == CameraType.SceneView)
            {
                return;
            }
            
            // 只有game窗口会应用renderPass
            renderer.EnqueuePass(m_ScriptablePass);
        }
    }
        
}


