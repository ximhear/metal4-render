//
//  Renderer.swift
//  Metal4Render
//
//  Metal 4 기반 3D 렌더링 엔진
//
//  이 파일은 Metal 4 API를 사용하여 왼손 좌표계(Left-Hand Coordinate System)와
//  원근 투영(Perspective Projection)을 구현한 렌더러를 포함합니다.
//
//  ┌─────────────────────────────────────────────────────────────────────────┐
//  │                        Metal 4 아키텍처 개요                              │
//  ├─────────────────────────────────────────────────────────────────────────┤
//  │                                                                         │
//  │   ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐       │
//  │   │   MTLDevice │───▶│ MTL4CommandQueue│───▶│ MTL4CommandBuffer│       │
//  │   └─────────────┘    └─────────────────┘    └──────────────────┘       │
//  │          │                                           │                  │
//  │          ▼                                           ▼                  │
//  │   ┌─────────────┐                          ┌──────────────────┐        │
//  │   │MTL4Compiler │                          │RenderCommandEncoder       │
//  │   └─────────────┘                          └──────────────────┘        │
//  │          │                                           │                  │
//  │          ▼                                           ▼                  │
//  │   ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐       │
//  │   │PipelineState│    │  ResidencySet   │    │  ArgumentTable   │       │
//  │   └─────────────┘    └─────────────────┘    └──────────────────┘       │
//  │                                                                         │
//  └─────────────────────────────────────────────────────────────────────────┘
//
//  Metal 4 vs 기존 Metal 비교:
//  ═══════════════════════════════════════════════════════════════════════════
//
//  ┌────────────────────┬─────────────────────┬─────────────────────────────┐
//  │      기능          │    기존 Metal        │        Metal 4              │
//  ├────────────────────┼─────────────────────┼─────────────────────────────┤
//  │ Command Queue      │ MTLCommandQueue     │ MTL4CommandQueue            │
//  │ Command Buffer     │ 매 프레임 새로 생성   │ 재사용 + CommandAllocator   │
//  │ Resource Binding   │ setVertexBuffer()   │ ArgumentTable + GPU Address │
//  │ Pipeline Creation  │ MTLDevice           │ MTL4Compiler                │
//  │ Memory Management  │ 암시적              │ ResidencySet (명시적)        │
//  │ Synchronization    │ waitUntilCompleted  │ SharedEvent + Signal        │
//  └────────────────────┴─────────────────────┴─────────────────────────────┘
//
//  파일 구조:
//  ─────────────────────────────────────────────────────────────────────────
//  1. 데이터 구조체 (Vertex, Uniforms)
//  2. Metal4Renderer - Metal 4 API 기반 렌더러 (macOS 26.0+, iOS 26.0+)
//  3. LegacyMetalRenderer - 기존 Metal API 기반 렌더러 (하위 호환성)
//

import MetalKit
import simd

// ════════════════════════════════════════════════════════════════════════════
// MARK: - 데이터 구조체 (Data Structures)
// ════════════════════════════════════════════════════════════════════════════

/// 버텍스(정점) 데이터 구조체
///
/// 3D 공간의 각 정점에 대한 위치와 색상 정보를 저장합니다.
/// GPU의 Metal 셰이더(`Shaders.metal`)와 메모리 레이아웃이 일치해야 합니다.
///
/// ## 메모리 레이아웃
///
/// ```
/// ┌─────────────────────────────────────────────────────────────┐
/// │ Offset 0-11    │ Offset 12-15  │ Offset 16-31              │
/// │ position       │ (padding)     │ color                     │
/// │ SIMD3<Float>   │               │ SIMD4<Float>              │
/// │ x, y, z        │               │ r, g, b, a                │
/// └─────────────────────────────────────────────────────────────┘
/// Total stride: 32 bytes (SIMD3는 16바이트로 정렬됨)
/// ```
///
/// ## 삼각형 정의 예시
///
/// ```
///            (0.0, 0.5, 0.0) 빨강
///                  ▲
///                 ╱ ╲
///                ╱   ╲
///               ╱     ╲
///              ╱       ╲
///             ╱         ╲
///            ▼───────────▼
/// (-0.5, -0.5, 0.0)   (0.5, -0.5, 0.0)
///       파랑                초록
/// ```
///
/// - Note: SIMD3<Float>는 12바이트지만 Metal은 16바이트 정렬을 사용할 수 있습니다.
///         `MemoryLayout<Vertex>.stride`를 사용하여 실제 크기를 확인하세요.
struct Vertex {
    /// 3D 공간에서의 버텍스 위치
    ///
    /// 왼손 좌표계 기준:
    /// - x: 양수 = 오른쪽, 음수 = 왼쪽
    /// - y: 양수 = 위쪽, 음수 = 아래쪽
    /// - z: 양수 = 화면 안쪽(멀어짐), 음수 = 화면 바깥쪽(가까워짐)
    var position: SIMD3<Float>

    /// 버텍스 색상 (RGBA)
    ///
    /// 각 컴포넌트는 0.0 ~ 1.0 범위:
    /// - x (r): 빨간색 강도
    /// - y (g): 초록색 강도
    /// - z (b): 파란색 강도
    /// - w (a): 알파(불투명도), 1.0 = 완전 불투명
    var color: SIMD4<Float>
}

/// 유니폼(Uniform) 데이터 구조체
///
/// 모든 버텍스에 동일하게 적용되는 변환 행렬들을 포함합니다.
/// 매 프레임마다 CPU에서 계산되어 GPU로 전달됩니다.
///
/// ## 변환 순서 (Transformation Pipeline)
///
/// ```
/// ┌──────────┐     Model      ┌──────────┐     View       ┌──────────┐
/// │  Local   │ ─────────────▶ │  World   │ ─────────────▶ │   Eye    │
/// │  Space   │    Matrix      │  Space   │    Matrix      │  Space   │
/// └──────────┘                └──────────┘                └──────────┘
///                                                               │
///                                                               │ Projection
///                                                               │ Matrix
///                                                               ▼
///                                                         ┌──────────┐
///                                                         │   Clip   │
///                                                         │  Space   │
///                                                         └──────────┘
/// ```
///
/// ## 행렬 곱셈 순서
///
/// ```swift
/// clipPosition = projectionMatrix × viewMatrix × modelMatrix × localPosition
/// ```
///
/// - Important: Metal은 column-major 행렬을 사용합니다.
struct Uniforms {
    /// 모델 변환 행렬
    ///
    /// 오브젝트의 로컬 좌표를 월드 좌표로 변환합니다.
    /// 이 프로젝트에서는 Y축 회전에 사용됩니다.
    var modelMatrix: float4x4

    /// 뷰 변환 행렬
    ///
    /// 월드 좌표를 카메라 기준 좌표로 변환합니다.
    /// `lookAtLH()` 함수로 생성됩니다.
    var viewMatrix: float4x4

    /// 투영 변환 행렬
    ///
    /// 3D 좌표를 2D 클립 좌표로 변환합니다.
    /// `perspectiveProjectionLH()` 함수로 생성됩니다.
    var projectionMatrix: float4x4
}


// ════════════════════════════════════════════════════════════════════════════
// MARK: - Metal 4 Renderer
// ════════════════════════════════════════════════════════════════════════════

/// Metal 4 기반 렌더러
///
/// WWDC 2025에서 발표된 Metal 4 API를 사용하여 3D 그래픽을 렌더링합니다.
/// 왼손 좌표계와 원근 투영을 사용하여 회전하는 삼각형을 표시합니다.
///
/// ## Metal 4의 주요 특징
///
/// ### 1. MTL4CommandQueue & MTL4CommandBuffer
/// ```
/// 기존 Metal:
/// ┌─────────────────────────────────────────────────────┐
/// │ 매 프레임마다 새로운 CommandBuffer 생성              │
/// │ commandQueue.makeCommandBuffer() → 사용 → 폐기      │
/// └─────────────────────────────────────────────────────┘
///
/// Metal 4:
/// ┌─────────────────────────────────────────────────────┐
/// │ CommandBuffer 재사용 + CommandAllocator             │
/// │ commandBuffer.beginCommandBuffer(allocator:)        │
/// │ ... 렌더링 명령 ...                                  │
/// │ commandBuffer.endCommandBuffer()                    │
/// └─────────────────────────────────────────────────────┘
/// ```
///
/// ### 2. MTL4ArgumentTable
/// ```
/// 기존 Metal:
///   renderEncoder.setVertexBuffer(buffer, offset: 0, index: 0)
///
/// Metal 4:
///   argumentTable.setAddress(buffer.gpuAddress, index: 0)
///   renderEncoder.setArgumentTable(argumentTable, stages: .vertex)
/// ```
///
/// ### 3. MTLResidencySet
/// ```
/// GPU 메모리 관리를 명시적으로 제어:
///   residencySet.addAllocation(buffer)
///   residencySet.commit()
///   commandQueue.addResidencySet(residencySet)
/// ```
///
/// ### 4. MTL4Compiler
/// ```
/// 파이프라인 상태 생성을 위한 새로운 컴파일러:
///   let compiler = device.makeCompiler(descriptor:)
///   let pipelineState = compiler.makeRenderPipelineState(descriptor:)
/// ```
///
/// ## 프레임 동기화 (Triple Buffering)
///
/// ```
/// Frame 0: [Render] [Wait GPU] [        ]
/// Frame 1: [      ] [Render  ] [Wait GPU]
/// Frame 2: [      ] [        ] [Render  ]
/// Frame 3: [Render] [        ] [        ]  ← Frame 0의 GPU 완료 후 재사용
/// ```
///
/// - Requires: macOS 26.0+, iOS 26.0+, visionOS 26.0+
@available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
class Metal4Renderer: NSObject, MTKViewDelegate {

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Properties
    // ════════════════════════════════════════════════════════════════════════

    /// Metal 디바이스
    ///
    /// GPU를 추상화한 객체입니다. 모든 Metal 리소스 생성의 시작점입니다.
    /// `MTLCreateSystemDefaultDevice()`로 시스템의 기본 GPU를 가져옵니다.
    let device: MTLDevice

    /// 동시 처리 가능한 최대 프레임 수 (Triple Buffering)
    ///
    /// CPU와 GPU가 병렬로 작업할 수 있도록 3개의 프레임 버퍼를 사용합니다.
    /// - Frame N: GPU에서 렌더링 중
    /// - Frame N+1: CPU에서 준비 중
    /// - Frame N+2: 대기 중
    let maxFramesInFlight: UInt64 = 3

    // ────────────────────────────────────────────────────────────────────────
    // Metal 4 Core Objects
    // ────────────────────────────────────────────────────────────────────────

    /// Metal 4 커맨드 큐
    ///
    /// GPU에 제출할 커맨드 버퍼들의 대기열입니다.
    /// Metal 4에서는 `MTL4CommandQueue`를 사용하여 더 효율적인
    /// 커맨드 제출과 동기화를 지원합니다.
    private let commandQueue: MTL4CommandQueue

    /// Metal 4 커맨드 버퍼 (재사용)
    ///
    /// Metal 4에서는 커맨드 버퍼를 재사용할 수 있습니다.
    /// `beginCommandBuffer(allocator:)` / `endCommandBuffer()`로
    /// 각 프레임의 시작과 끝을 표시합니다.
    private let commandBuffer: MTL4CommandBuffer

    /// 커맨드 할당자 배열 (프레임당 하나)
    ///
    /// 각 프레임의 커맨드 데이터를 저장하는 메모리 풀입니다.
    /// Triple buffering을 위해 3개의 할당자를 순환 사용합니다.
    ///
    /// ```
    /// Frame 0: allocators[0] → 사용 중
    /// Frame 1: allocators[1] → 사용 중
    /// Frame 2: allocators[2] → 사용 중
    /// Frame 3: allocators[0] → Frame 0 완료 후 재사용 (reset)
    /// ```
    private let commandAllocators: [MTL4CommandAllocator]

    /// Metal 4 컴파일러
    ///
    /// 셰이더 함수와 파이프라인 설정을 컴파일하여
    /// GPU에서 실행 가능한 파이프라인 상태를 생성합니다.
    private let compiler: MTL4Compiler

    /// 레지던시 셋 (GPU 메모리 관리)
    ///
    /// GPU가 접근해야 하는 리소스들을 명시적으로 등록합니다.
    /// Metal 4에서는 이 방식으로 메모리 관리를 더 세밀하게 제어할 수 있습니다.
    ///
    /// ```swift
    /// residencySet.addAllocation(vertexBuffer)
    /// residencySet.addAllocation(uniformBuffer)
    /// residencySet.commit()  // 변경사항 적용
    /// commandQueue.addResidencySet(residencySet)  // 커맨드 큐에 연결
    /// ```
    private let residencySet: MTLResidencySet

    /// 프레임 완료 이벤트
    ///
    /// CPU-GPU 간 동기화를 위한 공유 이벤트입니다.
    /// 각 프레임 완료 시 시그널되어 다음 프레임이
    /// 해당 리소스를 안전하게 재사용할 수 있음을 알립니다.
    private let frameCompletionEvent: MTLSharedEvent

    /// 버텍스 인자 테이블
    ///
    /// Metal 4의 새로운 리소스 바인딩 방식입니다.
    /// GPU 주소를 직접 설정하여 셰이더에서 접근할 수 있게 합니다.
    ///
    /// ```swift
    /// // 버퍼의 GPU 주소를 인덱스 0에 바인딩
    /// vertexArgumentTable.setAddress(vertexBuffer.gpuAddress, index: 0)
    ///
    /// // 렌더 인코더에 테이블 설정
    /// renderEncoder.setArgumentTable(vertexArgumentTable, stages: .vertex)
    /// ```
    private let vertexArgumentTable: MTL4ArgumentTable

    // ────────────────────────────────────────────────────────────────────────
    // Pipeline & Buffers
    // ────────────────────────────────────────────────────────────────────────

    /// 렌더 파이프라인 상태
    ///
    /// 버텍스 셰이더, 프래그먼트 셰이더, 픽셀 포맷 등
    /// 렌더링에 필요한 모든 설정이 컴파일된 상태입니다.
    private var pipelineState: MTLRenderPipelineState!

    /// 깊이 스텐실 상태
    ///
    /// 깊이 테스트(Depth Test) 설정입니다.
    /// 가까운 물체가 먼 물체를 가리도록 합니다.
    private var depthStencilState: MTLDepthStencilState!

    /// 버텍스 버퍼
    ///
    /// 삼각형의 3개 버텍스 데이터가 저장된 GPU 버퍼입니다.
    private var vertexBuffer: MTLBuffer!

    /// 유니폼 버퍼 배열 (프레임당 하나)
    ///
    /// 각 프레임의 변환 행렬을 저장합니다.
    /// Triple buffering을 위해 3개를 순환 사용합니다.
    private var uniformBuffers: [MTLBuffer]!

    // ────────────────────────────────────────────────────────────────────────
    // Animation State
    // ────────────────────────────────────────────────────────────────────────

    /// 현재 프레임 인덱스
    ///
    /// 매 프레임마다 1씩 증가합니다.
    /// `frameIndex % maxFramesInFlight`로 현재 사용할 버퍼 인덱스를 계산합니다.
    private var frameIndex: UInt64 = 0

    /// 현재 회전 각도 (라디안)
    ///
    /// 매 프레임마다 증가하여 삼각형이 Y축을 중심으로 회전하게 합니다.
    private var rotation: Float = 0

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Initialization
    // ════════════════════════════════════════════════════════════════════════

    /// 렌더러 초기화
    ///
    /// Metal 디바이스, 커맨드 큐, 파이프라인 등 모든 렌더링 리소스를 생성합니다.
    ///
    /// ## 초기화 순서
    ///
    /// ```
    /// 1. MTLDevice 획득
    /// 2. MTL4CommandQueue 생성
    /// 3. MTL4CommandBuffer 생성
    /// 4. MTL4CommandAllocator 배열 생성 (Triple Buffering)
    /// 5. MTL4Compiler 생성
    /// 6. MTLResidencySet 생성
    /// 7. MTLSharedEvent 생성 (동기화용)
    /// 8. MTL4ArgumentTable 생성
    /// 9. MTKView 설정
    /// 10. 버텍스/유니폼 버퍼 생성
    /// 11. 파이프라인 상태 컴파일
    /// 12. 레지던시 셋 설정
    /// ```
    ///
    /// - Parameter mtkView: 렌더링 결과를 표시할 MetalKit 뷰
    /// - Returns: 초기화된 렌더러, 실패 시 nil
    init?(mtkView: MTKView) {
        // ────────────────────────────────────────────────────────────────────
        // 1. Metal Device 획득
        // ────────────────────────────────────────────────────────────────────
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ Metal을 지원하지 않는 디바이스입니다.")
            return nil
        }
        self.device = device

        // ────────────────────────────────────────────────────────────────────
        // 2. Metal 4 Command Queue 생성
        // ────────────────────────────────────────────────────────────────────
        // MTL4CommandQueue는 기존 MTLCommandQueue와 달리
        // 더 세밀한 동기화 제어와 효율적인 커맨드 제출을 지원합니다.
        guard let commandQueue = device.makeMTL4CommandQueue() else {
            print("❌ MTL4CommandQueue 생성 실패")
            return nil
        }
        self.commandQueue = commandQueue

        // ────────────────────────────────────────────────────────────────────
        // 3. Metal 4 Command Buffer 생성
        // ────────────────────────────────────────────────────────────────────
        // Metal 4에서는 커맨드 버퍼를 재사용할 수 있습니다.
        // beginCommandBuffer/endCommandBuffer로 각 프레임을 구분합니다.
        guard let commandBuffer = device.makeCommandBuffer() else {
            print("❌ MTL4CommandBuffer 생성 실패")
            return nil
        }
        self.commandBuffer = commandBuffer

        // ────────────────────────────────────────────────────────────────────
        // 4. Command Allocator 배열 생성 (Triple Buffering)
        // ────────────────────────────────────────────────────────────────────
        // 각 프레임의 커맨드 데이터를 저장할 메모리 풀입니다.
        // 프레임 N이 완료되면 해당 할당자를 reset하여 재사용합니다.
        self.commandAllocators = (0..<maxFramesInFlight).compactMap { _ in
            device.makeCommandAllocator()
        }
        guard commandAllocators.count == Int(maxFramesInFlight) else {
            print("❌ CommandAllocator 생성 실패")
            return nil
        }

        // ────────────────────────────────────────────────────────────────────
        // 5. Metal 4 Compiler 생성
        // ────────────────────────────────────────────────────────────────────
        // 셰이더 함수를 파이프라인 상태로 컴파일합니다.
        let compilerDescriptor = MTL4CompilerDescriptor()
        guard let compiler = try? device.makeCompiler(descriptor: compilerDescriptor) else {
            print("❌ MTL4Compiler 생성 실패")
            return nil
        }
        self.compiler = compiler

        // ────────────────────────────────────────────────────────────────────
        // 6. Residency Set 생성
        // ────────────────────────────────────────────────────────────────────
        // GPU가 접근할 리소스들을 명시적으로 등록합니다.
        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.initialCapacity = 16  // 예상 리소스 수
        guard let residencySet = try? device.makeResidencySet(descriptor: residencyDescriptor) else {
            print("❌ ResidencySet 생성 실패")
            return nil
        }
        self.residencySet = residencySet

        // ────────────────────────────────────────────────────────────────────
        // 7. Frame Completion Event 생성
        // ────────────────────────────────────────────────────────────────────
        // CPU-GPU 간 동기화를 위한 공유 이벤트입니다.
        guard let event = device.makeSharedEvent() else {
            print("❌ SharedEvent 생성 실패")
            return nil
        }
        self.frameCompletionEvent = event

        // ────────────────────────────────────────────────────────────────────
        // 8. Argument Table 생성
        // ────────────────────────────────────────────────────────────────────
        // Metal 4의 새로운 리소스 바인딩 방식입니다.
        let argumentDescriptor = MTL4ArgumentTableDescriptor()
        argumentDescriptor.maxBufferBindCount = 2  // vertices(0) + uniforms(1)
        guard let vertexArgumentTable = try? device.makeArgumentTable(descriptor: argumentDescriptor) else {
            print("❌ ArgumentTable 생성 실패")
            return nil
        }
        self.vertexArgumentTable = vertexArgumentTable

        super.init()

        // ────────────────────────────────────────────────────────────────────
        // 9. MTKView 설정
        // ────────────────────────────────────────────────────────────────────
        mtkView.device = device
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        mtkView.colorPixelFormat = .bgra8Unorm_srgb  // sRGB 색공간
        mtkView.depthStencilPixelFormat = .depth32Float  // 32비트 깊이 버퍼

        // 이벤트 초기값 설정
        frameCompletionEvent.signaledValue = frameIndex

        // ────────────────────────────────────────────────────────────────────
        // 10-12. 리소스 생성 및 설정
        // ────────────────────────────────────────────────────────────────────
        buildResources()
        buildPipelineState()
        setupResidency()
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Resource Building
    // ════════════════════════════════════════════════════════════════════════

    /// GPU 리소스 생성
    ///
    /// 버텍스 버퍼, 유니폼 버퍼, 깊이 스텐실 상태를 생성합니다.
    ///
    /// ## 버텍스 데이터
    ///
    /// ```
    ///            Red (0.0, 0.5, 0.0)
    ///                  ▲
    ///                 ╱ ╲
    ///                ╱   ╲
    ///               ╱     ╲
    ///              ╱       ╲
    ///             ╱         ╲
    ///            ▼───────────▼
    ///   Blue (-0.5, -0.5)   Green (0.5, -0.5)
    /// ```
    private func buildResources() {
        // ────────────────────────────────────────────────────────────────────
        // 삼각형 버텍스 정의
        // ────────────────────────────────────────────────────────────────────
        // 왼손 좌표계(Left-Hand Coordinate System):
        // - +X: 오른쪽
        // - +Y: 위쪽
        // - +Z: 화면 안쪽 (멀어지는 방향)
        //
        // 버텍스 순서: 시계 방향 (Counter-Clockwise에서 바라볼 때)
        // Front-face culling을 위해 setFrontFacing(.counterClockwise) 설정 필요
        let vertices: [Vertex] = [
            // 상단 - 빨간색
            Vertex(position: SIMD3<Float>(0.0, 0.5, 0.0),
                   color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0)),

            // 우하단 - 초록색
            Vertex(position: SIMD3<Float>(0.5, -0.5, 0.0),
                   color: SIMD4<Float>(0.0, 1.0, 0.0, 1.0)),

            // 좌하단 - 파란색
            Vertex(position: SIMD3<Float>(-0.5, -0.5, 0.0),
                   color: SIMD4<Float>(0.0, 0.0, 1.0, 1.0))
        ]

        // ────────────────────────────────────────────────────────────────────
        // 버텍스 버퍼 생성
        // ────────────────────────────────────────────────────────────────────
        // .storageModeShared: CPU와 GPU 모두 접근 가능
        // 작은 데이터에 적합하며, 매 프레임 업데이트가 필요한 경우에도 사용
        vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<Vertex>.stride * vertices.count,
            options: .storageModeShared
        )

        // ────────────────────────────────────────────────────────────────────
        // 유니폼 버퍼 배열 생성 (Triple Buffering)
        // ────────────────────────────────────────────────────────────────────
        // 각 프레임이 독립적인 버퍼를 사용하여
        // CPU 업데이트와 GPU 읽기가 충돌하지 않도록 합니다.
        uniformBuffers = (0..<maxFramesInFlight).map { _ in
            device.makeBuffer(
                length: MemoryLayout<Uniforms>.stride,
                options: .storageModeShared
            )!
        }

        // ────────────────────────────────────────────────────────────────────
        // 깊이 스텐실 상태 생성
        // ────────────────────────────────────────────────────────────────────
        // 깊이 테스트: 새 픽셀의 깊이가 기존보다 작으면(가까우면) 그리기
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.isDepthWriteEnabled = true      // 깊이 버퍼 쓰기 활성화
        depthDescriptor.depthCompareFunction = .less    // 더 가까운 것만 그리기
        depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
    }

    /// 렌더 파이프라인 상태 생성
    ///
    /// 버텍스 셰이더와 프래그먼트 셰이더를 컴파일하고
    /// 렌더링 설정을 하나의 파이프라인 상태로 결합합니다.
    ///
    /// ## Metal 4 파이프라인 생성 과정
    ///
    /// ```
    /// ┌─────────────────────────────────────────────────────────────────────┐
    /// │                                                                     │
    /// │  MTLLibrary ──▶ MTL4LibraryFunctionDescriptor ──┐                  │
    /// │  (Shaders.metal)           (vertexShader)       │                  │
    /// │                                                 ▼                  │
    /// │                            MTL4RenderPipelineDescriptor            │
    /// │                                                 │                  │
    /// │  MTLLibrary ──▶ MTL4LibraryFunctionDescriptor ──┘                  │
    /// │                         (fragmentShader)                           │
    /// │                                                 │                  │
    /// │                                                 ▼                  │
    /// │                             MTL4Compiler                           │
    /// │                                                 │                  │
    /// │                                                 ▼                  │
    /// │                         MTLRenderPipelineState                     │
    /// │                                                                     │
    /// └─────────────────────────────────────────────────────────────────────┘
    /// ```
    private func buildPipelineState() {
        // ────────────────────────────────────────────────────────────────────
        // 셰이더 라이브러리 로드
        // ────────────────────────────────────────────────────────────────────
        guard let library = device.makeDefaultLibrary() else {
            fatalError("❌ 기본 Metal 라이브러리를 찾을 수 없습니다. Shaders.metal 파일을 확인하세요.")
        }

        // ────────────────────────────────────────────────────────────────────
        // Metal 4 방식의 함수 디스크립터 생성
        // ────────────────────────────────────────────────────────────────────
        // MTL4LibraryFunctionDescriptor: 셰이더 함수를 참조하는 Metal 4 방식
        let vertexFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        vertexFunctionDescriptor.name = "vertexShader"
        vertexFunctionDescriptor.library = library

        let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        fragmentFunctionDescriptor.name = "fragmentShader"
        fragmentFunctionDescriptor.library = library

        // ────────────────────────────────────────────────────────────────────
        // 파이프라인 디스크립터 설정
        // ────────────────────────────────────────────────────────────────────
        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.vertexFunctionDescriptor = vertexFunctionDescriptor
        pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunctionDescriptor

        // 색상 첨부 설정 (렌더 타겟)
        // .bgra8Unorm_srgb: 8비트 BGRA, sRGB 색공간 (감마 보정 적용)
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb

        // ────────────────────────────────────────────────────────────────────
        // Metal 4 Compiler로 파이프라인 상태 생성
        // ────────────────────────────────────────────────────────────────────
        do {
            pipelineState = try compiler.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("❌ 파이프라인 상태 생성 실패: \(error)")
        }
    }

    /// 레지던시 셋 설정
    ///
    /// GPU가 접근해야 하는 모든 리소스를 등록합니다.
    /// Metal 4에서는 이 과정이 필수입니다.
    private func setupResidency() {
        // 버텍스 버퍼 등록
        residencySet.addAllocation(vertexBuffer)

        // 모든 유니폼 버퍼 등록
        for buffer in uniformBuffers {
            residencySet.addAllocation(buffer)
        }

        // 변경사항 커밋
        residencySet.commit()

        // 커맨드 큐에 레지던시 셋 연결
        commandQueue.addResidencySet(residencySet)
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Matrix Functions (Left-Hand Coordinate System)
    // ════════════════════════════════════════════════════════════════════════

    /// 왼손 좌표계 원근 투영 행렬 생성
    ///
    /// 3D 뷰 좌표를 2D 클립 좌표로 변환하는 행렬을 생성합니다.
    /// 멀리 있는 물체가 작게 보이는 원근감을 구현합니다.
    ///
    /// ## 원근 투영 원리
    ///
    /// ```
    ///                    Near Plane              Far Plane
    ///                        │                       │
    ///                   ┌────┼────┐             ┌────┼────┐
    ///                   │    │    │             │    │    │
    ///       Eye ════════╪════════════════════════════╪════▶ +Z
    ///       (카메라)     │    │    │             │    │    │
    ///                   └────┼────┘             └────┼────┘
    ///                        │                       │
    ///                   (크게 보임)              (작게 보임)
    ///
    ///       ◀──── FOV (Field of View) ────▶
    /// ```
    ///
    /// ## 왼손 좌표계 특징
    ///
    /// - Z축 양의 방향이 화면 안쪽 (카메라가 바라보는 방향)
    /// - 깊이 값: 0 (near) ~ 1 (far)
    /// - DirectX, Unity 등에서 사용
    ///
    /// ## 행렬 구조
    ///
    /// ```
    /// ┌                                        ┐
    /// │ xScale    0        0              0    │
    /// │   0    yScale      0              0    │
    /// │   0       0    far/(far-near)     1    │  ← w = z (원근 분할용)
    /// │   0       0   -near*far/(far-near) 0   │
    /// └                                        ┘
    ///
    /// 여기서:
    /// - yScale = 1 / tan(fovy / 2)
    /// - xScale = yScale / aspect
    /// ```
    ///
    /// - Parameters:
    ///   - fovy: 수직 시야각 (라디안). 일반적으로 45° ~ 90° (π/4 ~ π/2)
    ///   - aspect: 화면 종횡비 (width / height)
    ///   - nearZ: 근평면 거리 (이보다 가까운 물체는 클리핑)
    ///   - farZ: 원평면 거리 (이보다 먼 물체는 클리핑)
    /// - Returns: 4x4 원근 투영 행렬
    ///
    /// - Important: nearZ는 0보다 커야 합니다. 0에 가까울수록 깊이 정밀도가 떨어집니다.
    func perspectiveProjectionLH(fovYRadians fovy: Float,
                                  aspect: Float,
                                  nearZ: Float,
                                  farZ: Float) -> float4x4 {
        // 수직 스케일: FOV가 클수록 더 많이 보임 (스케일 감소)
        let yScale = 1 / tan(fovy * 0.5)

        // 수평 스케일: 종횡비 보정
        let xScale = yScale / aspect

        // 깊이 범위
        let zRange = farZ - nearZ

        // Column-major 순서로 행렬 생성
        return float4x4(columns: (
            SIMD4<Float>(xScale, 0, 0, 0),                      // 열 0: X 스케일
            SIMD4<Float>(0, yScale, 0, 0),                      // 열 1: Y 스케일
            SIMD4<Float>(0, 0, farZ / zRange, 1),               // 열 2: Z 매핑 + W=Z
            SIMD4<Float>(0, 0, -nearZ * farZ / zRange, 0)       // 열 3: Z 오프셋
        ))
    }

    /// 왼손 좌표계 Look-At 뷰 행렬 생성
    ///
    /// 카메라의 위치와 방향을 기반으로 월드 좌표를 뷰 좌표로 변환하는 행렬을 생성합니다.
    ///
    /// ## Look-At 개념
    ///
    /// ```
    ///                          target (0, 0, 0)
    ///                               ●
    ///                              ╱│
    ///                             ╱ │
    ///                            ╱  │ up
    ///                           ╱   │
    ///                          ╱    │
    ///                         ╱     │
    ///              eye ●─────╱──────┘
    ///           (0, 0, -3)
    ///                    └── 카메라가 target을 바라봄
    /// ```
    ///
    /// ## 좌표축 계산
    ///
    /// ```
    /// 왼손 좌표계에서:
    /// 1. zAxis (forward) = normalize(target - eye)   // 카메라 → 타겟
    /// 2. xAxis (right)   = normalize(cross(up, zAxis))
    /// 3. yAxis (up)      = cross(zAxis, xAxis)
    /// ```
    ///
    /// ## 결과 행렬
    ///
    /// ```
    /// ┌                                                    ┐
    /// │  xAxis.x   yAxis.x   zAxis.x   0                   │
    /// │  xAxis.y   yAxis.y   zAxis.y   0                   │
    /// │  xAxis.z   yAxis.z   zAxis.z   0                   │
    /// │ -dot(x,e) -dot(y,e) -dot(z,e)  1                   │
    /// └                                                    ┘
    /// 여기서 e = eye (카메라 위치)
    /// ```
    ///
    /// - Parameters:
    ///   - eye: 카메라 위치
    ///   - target: 카메라가 바라보는 지점
    ///   - up: 카메라의 위쪽 방향 (보통 (0, 1, 0))
    /// - Returns: 4x4 뷰 변환 행렬
    func lookAtLH(eye: SIMD3<Float>,
                  target: SIMD3<Float>,
                  up: SIMD3<Float>) -> float4x4 {
        // 왼손 좌표계: 카메라는 +Z 방향을 바라봄
        let zAxis = normalize(target - eye)           // 전방 벡터
        let xAxis = normalize(cross(up, zAxis))       // 우측 벡터
        let yAxis = cross(zAxis, xAxis)               // 상향 벡터

        // 뷰 행렬 = 회전 × 이동 (카메라를 원점으로 이동)
        return float4x4(columns: (
            SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0),
            SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0),
            SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0),
            SIMD4<Float>(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1)
        ))
    }

    /// Y축 회전 행렬 생성
    ///
    /// 오브젝트를 Y축(수직축)을 중심으로 회전시키는 행렬을 생성합니다.
    ///
    /// ## 회전 방향
    ///
    /// 왼손 좌표계에서 양의 각도는 위에서 볼 때 시계 방향입니다.
    ///
    /// ```
    ///       +Y (위에서 내려다 봄)
    ///         │
    ///         │    θ > 0
    ///         │   ↻ (시계 방향)
    ///         │
    ///    ─────┼─────▶ +X
    ///         │
    ///         │
    ///         ▼
    ///        +Z
    /// ```
    ///
    /// ## 회전 행렬
    ///
    /// ```
    /// ┌                    ┐
    /// │  cos(θ)  0  -sin(θ)  0 │
    /// │    0     1     0     0 │
    /// │  sin(θ)  0   cos(θ)  0 │
    /// │    0     0     0     1 │
    /// └                    ┘
    /// ```
    ///
    /// - Parameter angle: 회전 각도 (라디안)
    /// - Returns: 4x4 Y축 회전 행렬
    func rotationY(_ angle: Float) -> float4x4 {
        let c = cos(angle)
        let s = sin(angle)

        return float4x4(columns: (
            SIMD4<Float>(c, 0, -s, 0),    // 열 0
            SIMD4<Float>(0, 1, 0, 0),     // 열 1 (Y축은 변화 없음)
            SIMD4<Float>(s, 0, c, 0),     // 열 2
            SIMD4<Float>(0, 0, 0, 1)      // 열 3 (이동 없음)
        ))
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - MTKViewDelegate
    // ════════════════════════════════════════════════════════════════════════

    /// 뷰 크기 변경 콜백
    ///
    /// 창 크기가 변경되거나 화면 회전 시 호출됩니다.
    /// 투영 행렬의 종횡비가 draw(in:)에서 매 프레임 계산되므로
    /// 여기서는 별도 처리가 필요 없습니다.
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // 필요시 렌더 타겟 재생성 등의 작업 수행
    }

    /// 프레임 렌더링
    ///
    /// 매 프레임마다 호출되어 3D 씬을 렌더링합니다.
    ///
    /// ## 렌더링 루프 상세
    ///
    /// ```
    /// ┌─────────────────────────────────────────────────────────────────────┐
    /// │                         Frame N 렌더링                               │
    /// ├─────────────────────────────────────────────────────────────────────┤
    /// │                                                                     │
    /// │  1. 이전 프레임 완료 대기 (N - maxFramesInFlight)                    │
    /// │     └─▶ frameCompletionEvent.wait()                                │
    /// │                                                                     │
    /// │  2. 유니폼 데이터 업데이트                                           │
    /// │     ├─▶ Model Matrix (회전)                                         │
    /// │     ├─▶ View Matrix (카메라)                                        │
    /// │     └─▶ Projection Matrix (원근)                                    │
    /// │                                                                     │
    /// │  3. Command Allocator 리셋                                          │
    /// │     └─▶ allocators[frameIndex % 3].reset()                         │
    /// │                                                                     │
    /// │  4. Command Buffer 시작                                             │
    /// │     └─▶ commandBuffer.beginCommandBuffer(allocator:)               │
    /// │                                                                     │
    /// │  5. Render Pass 실행                                                │
    /// │     ├─▶ makeRenderCommandEncoder()                                  │
    /// │     ├─▶ setArgumentTable() - 버퍼 바인딩                            │
    /// │     ├─▶ drawPrimitives() - 삼각형 그리기                            │
    /// │     └─▶ endEncoding()                                               │
    /// │                                                                     │
    /// │  6. Command Buffer 종료 및 제출                                      │
    /// │     ├─▶ commandBuffer.endCommandBuffer()                            │
    /// │     ├─▶ commandQueue.waitForDrawable()                              │
    /// │     ├─▶ commandQueue.commit([commandBuffer])                        │
    /// │     └─▶ commandQueue.signalDrawable()                               │
    /// │                                                                     │
    /// │  7. 프레임 완료 시그널                                               │
    /// │     └─▶ commandQueue.signalEvent(frameCompletionEvent)              │
    /// │                                                                     │
    /// └─────────────────────────────────────────────────────────────────────┘
    /// ```
    func draw(in view: MTKView) {
        // ────────────────────────────────────────────────────────────────────
        // Metal 4 Render Pass Descriptor 획득
        // ────────────────────────────────────────────────────────────────────
        guard let renderPassDescriptor = view.currentMTL4RenderPassDescriptor,
              let drawable = view.currentDrawable else {
            return
        }

        // ────────────────────────────────────────────────────────────────────
        // 1. 이전 프레임 완료 대기 (Triple Buffering 동기화)
        // ────────────────────────────────────────────────────────────────────
        // Frame N을 렌더링하려면 Frame N-3이 완료되어야 합니다.
        // 이를 통해 해당 프레임의 리소스(유니폼 버퍼 등)를 안전하게 재사용할 수 있습니다.
        if frameIndex >= maxFramesInFlight {
            let valueToWait = frameIndex - maxFramesInFlight
            frameCompletionEvent.wait(untilSignaledValue: valueToWait, timeoutMS: 8)
        }

        // ────────────────────────────────────────────────────────────────────
        // 2. 애니메이션 업데이트
        // ────────────────────────────────────────────────────────────────────
        rotation += 0.02  // 매 프레임 약 1.1도 회전

        // ────────────────────────────────────────────────────────────────────
        // 3. 변환 행렬 계산
        // ────────────────────────────────────────────────────────────────────
        let aspect = Float(view.drawableSize.width / view.drawableSize.height)

        // Model Matrix: Y축 회전
        let modelMatrix = rotationY(rotation)

        // View Matrix: 카메라 설정
        // - 카메라 위치: (0, 0, -3) - 삼각형 앞에 위치
        // - 바라보는 점: (0, 0, 0) - 원점
        // - 위쪽 방향: (0, 1, 0) - Y축이 위
        let eye = SIMD3<Float>(0, 0, -3)
        let target = SIMD3<Float>(0, 0, 0)
        let up = SIMD3<Float>(0, 1, 0)
        let viewMatrix = lookAtLH(eye: eye, target: target, up: up)

        // Projection Matrix: 원근 투영
        // - FOV: 45도 (π/4 라디안)
        // - Near plane: 0.1
        // - Far plane: 100
        let projectionMatrix = perspectiveProjectionLH(
            fovYRadians: Float.pi / 4,
            aspect: aspect,
            nearZ: 0.1,
            farZ: 100
        )

        // ────────────────────────────────────────────────────────────────────
        // 4. 유니폼 버퍼 업데이트
        // ────────────────────────────────────────────────────────────────────
        var uniforms = Uniforms(
            modelMatrix: modelMatrix,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix
        )

        let currentUniformBuffer = uniformBuffers[Int(frameIndex % maxFramesInFlight)]
        currentUniformBuffer.contents().copyMemory(
            from: &uniforms,
            byteCount: MemoryLayout<Uniforms>.stride
        )

        // ────────────────────────────────────────────────────────────────────
        // 5. Command Allocator 리셋 및 Command Buffer 시작
        // ────────────────────────────────────────────────────────────────────
        frameIndex += 1
        let allocator = commandAllocators[Int(frameIndex % maxFramesInFlight)]
        allocator.reset()  // 이전 프레임의 커맨드 데이터 해제

        commandBuffer.beginCommandBuffer(allocator: allocator)

        // ────────────────────────────────────────────────────────────────────
        // 6. Render Command Encoder 생성 및 설정
        // ────────────────────────────────────────────────────────────────────
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

        // Front-face 설정 (CCW = Counter-Clockwise)
        renderEncoder.setFrontFacing(.counterClockwise)

        // 깊이 테스트 설정
        renderEncoder.setDepthStencilState(depthStencilState)

        // 파이프라인 상태 설정
        renderEncoder.setRenderPipelineState(pipelineState)

        // ────────────────────────────────────────────────────────────────────
        // 7. Metal 4 Argument Table을 통한 리소스 바인딩
        // ────────────────────────────────────────────────────────────────────
        // 기존 Metal: renderEncoder.setVertexBuffer(buffer, offset:, index:)
        // Metal 4: argumentTable.setAddress(buffer.gpuAddress, index:)
        //          renderEncoder.setArgumentTable(table, stages:)
        vertexArgumentTable.setAddress(vertexBuffer.gpuAddress, index: 0)
        vertexArgumentTable.setAddress(currentUniformBuffer.gpuAddress, index: 1)
        renderEncoder.setArgumentTable(vertexArgumentTable, stages: .vertex)

        // ────────────────────────────────────────────────────────────────────
        // 8. 삼각형 그리기
        // ────────────────────────────────────────────────────────────────────
        renderEncoder.drawPrimitives(
            primitiveType: .triangle,
            vertexStart: 0,
            vertexCount: 3
        )

        renderEncoder.endEncoding()

        // ────────────────────────────────────────────────────────────────────
        // 9. Command Buffer 종료 및 제출
        // ────────────────────────────────────────────────────────────────────
        commandBuffer.endCommandBuffer()

        // Metal 4 방식의 Drawable 동기화 및 제출
        commandQueue.waitForDrawable(drawable)    // Drawable 준비 대기
        commandQueue.commit([commandBuffer])       // GPU에 커맨드 제출
        commandQueue.signalDrawable(drawable)      // Drawable 완료 시그널
        drawable.present()                         // 화면에 표시

        // ────────────────────────────────────────────────────────────────────
        // 10. 프레임 완료 시그널
        // ────────────────────────────────────────────────────────────────────
        // 다음 프레임에서 이 프레임의 리소스를 재사용할 수 있음을 알림
        commandQueue.signalEvent(frameCompletionEvent, value: frameIndex)
    }
}


// ════════════════════════════════════════════════════════════════════════════
// MARK: - Legacy Metal Renderer (Fallback)
// ════════════════════════════════════════════════════════════════════════════

/// 기존 Metal API 기반 렌더러 (하위 호환성)
///
/// macOS 26.0 미만, iOS 26.0 미만 시스템에서 사용됩니다.
/// Metal 4 API 대신 기존 Metal API를 사용하여 동일한 결과를 렌더링합니다.
///
/// ## Metal 4와의 주요 차이점
///
/// | 기능 | Legacy Metal | Metal 4 |
/// |------|--------------|---------|
/// | Command Queue | MTLCommandQueue | MTL4CommandQueue |
/// | Command Buffer | 매 프레임 생성 | 재사용 + Allocator |
/// | Resource Binding | setVertexBuffer() | ArgumentTable |
/// | Pipeline | device.makeRenderPipelineState() | compiler.makeRenderPipelineState() |
/// | Memory | 암시적 관리 | ResidencySet |
///
/// - Note: 이 렌더러는 Metal 4가 지원되지 않는 환경에서 폴백으로 사용됩니다.
class LegacyMetalRenderer: NSObject, MTKViewDelegate {

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Properties
    // ════════════════════════════════════════════════════════════════════════

    /// Metal 디바이스
    let device: MTLDevice

    /// 커맨드 큐 (기존 Metal API)
    let commandQueue: MTLCommandQueue

    /// 렌더 파이프라인 상태
    var pipelineState: MTLRenderPipelineState!

    /// 버텍스 버퍼
    var vertexBuffer: MTLBuffer!

    /// 유니폼 버퍼 (단일 - 동기 렌더링)
    var uniformBuffer: MTLBuffer!

    /// 현재 회전 각도
    var rotation: Float = 0

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Initialization
    // ════════════════════════════════════════════════════════════════════════

    /// 레거시 렌더러 초기화
    ///
    /// - Parameter mtkView: 렌더링 결과를 표시할 MetalKit 뷰
    /// - Returns: 초기화된 렌더러, 실패 시 nil
    init?(mtkView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue

        mtkView.device = device
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        mtkView.depthStencilPixelFormat = .depth32Float

        super.init()

        buildPipelineState(mtkView: mtkView)
        buildBuffers()
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Resource Building
    // ════════════════════════════════════════════════════════════════════════

    /// 파이프라인 상태 생성 (기존 Metal 방식)
    private func buildPipelineState(mtkView: MTKView) {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("❌ 기본 Metal 라이브러리를 찾을 수 없습니다.")
        }

        // 기존 Metal 방식: makeFunction()으로 직접 함수 획득
        let vertexFunction = library.makeFunction(name: "vertexShader")
        let fragmentFunction = library.makeFunction(name: "fragmentShader")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = mtkView.depthStencilPixelFormat

        // 기존 Metal 방식: device에서 직접 파이프라인 생성
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("❌ 파이프라인 상태 생성 실패: \(error)")
        }
    }

    /// GPU 버퍼 생성
    private func buildBuffers() {
        // 삼각형 버텍스 데이터
        let vertices: [Vertex] = [
            Vertex(position: SIMD3<Float>(0.0, 0.5, 0.0), color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0)),
            Vertex(position: SIMD3<Float>(0.5, -0.5, 0.0), color: SIMD4<Float>(0.0, 1.0, 0.0, 1.0)),
            Vertex(position: SIMD3<Float>(-0.5, -0.5, 0.0), color: SIMD4<Float>(0.0, 0.0, 1.0, 1.0))
        ]

        vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<Vertex>.stride * vertices.count,
            options: .storageModeShared
        )

        uniformBuffer = device.makeBuffer(
            length: MemoryLayout<Uniforms>.stride,
            options: .storageModeShared
        )
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Matrix Functions
    // ════════════════════════════════════════════════════════════════════════

    /// 왼손 좌표계 원근 투영 행렬 (Metal4Renderer와 동일)
    func perspectiveProjectionLH(fovYRadians fovy: Float,
                                  aspect: Float,
                                  nearZ: Float,
                                  farZ: Float) -> float4x4 {
        let yScale = 1 / tan(fovy * 0.5)
        let xScale = yScale / aspect
        let zRange = farZ - nearZ

        return float4x4(columns: (
            SIMD4<Float>(xScale, 0, 0, 0),
            SIMD4<Float>(0, yScale, 0, 0),
            SIMD4<Float>(0, 0, farZ / zRange, 1),
            SIMD4<Float>(0, 0, -nearZ * farZ / zRange, 0)
        ))
    }

    /// 왼손 좌표계 Look-At 뷰 행렬 (Metal4Renderer와 동일)
    func lookAtLH(eye: SIMD3<Float>,
                  target: SIMD3<Float>,
                  up: SIMD3<Float>) -> float4x4 {
        let zAxis = normalize(target - eye)
        let xAxis = normalize(cross(up, zAxis))
        let yAxis = cross(zAxis, xAxis)

        return float4x4(columns: (
            SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0),
            SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0),
            SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0),
            SIMD4<Float>(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1)
        ))
    }

    /// Y축 회전 행렬 (Metal4Renderer와 동일)
    func rotationY(_ angle: Float) -> float4x4 {
        let c = cos(angle)
        let s = sin(angle)

        return float4x4(columns: (
            SIMD4<Float>(c, 0, -s, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(s, 0, c, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - MTKViewDelegate
    // ════════════════════════════════════════════════════════════════════════

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    /// 프레임 렌더링 (기존 Metal 방식)
    func draw(in view: MTKView) {
        // 렌더 패스 및 커맨드 버퍼 획득
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        // 회전 업데이트
        rotation += 0.01

        // 변환 행렬 계산
        let aspect = Float(view.drawableSize.width / view.drawableSize.height)
        let modelMatrix = rotationY(rotation)

        let eye = SIMD3<Float>(0, 0, -3)
        let target = SIMD3<Float>(0, 0, 0)
        let up = SIMD3<Float>(0, 1, 0)
        let viewMatrix = lookAtLH(eye: eye, target: target, up: up)

        let projectionMatrix = perspectiveProjectionLH(
            fovYRadians: Float.pi / 4,
            aspect: aspect,
            nearZ: 0.1,
            farZ: 100
        )

        // 유니폼 버퍼 업데이트
        var uniforms = Uniforms(
            modelMatrix: modelMatrix,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix
        )
        uniformBuffer.contents().copyMemory(from: &uniforms, byteCount: MemoryLayout<Uniforms>.stride)

        // 렌더링
        renderEncoder.setRenderPipelineState(pipelineState)

        // 기존 Metal 방식: setVertexBuffer()
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        renderEncoder.endEncoding()

        // 기존 Metal 방식: present() 후 commit()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
