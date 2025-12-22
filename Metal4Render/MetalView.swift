//
//  MetalView.swift
//  Metal4Render
//
//  SwiftUI와 Metal을 연결하는 브릿지 뷰
//
//  이 파일은 MetalKit의 MTKView를 SwiftUI에서 사용할 수 있도록
//  래핑(Wrapping)하는 역할을 합니다.
//
//  ┌─────────────────────────────────────────────────────────────────────────┐
//  │                      SwiftUI - Metal 통합 아키텍처                        │
//  ├─────────────────────────────────────────────────────────────────────────┤
//  │                                                                         │
//  │   ┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐    │
//  │   │  SwiftUI    │      │   MetalView     │      │    MTKView      │    │
//  │   │ ContentView │ ───▶ │ (Representable) │ ───▶ │   (MetalKit)    │    │
//  │   └─────────────┘      └─────────────────┘      └─────────────────┘    │
//  │                               │                         │              │
//  │                               ▼                         ▼              │
//  │                        ┌─────────────┐          ┌─────────────────┐    │
//  │                        │ Coordinator │ ───────▶ │    Renderer     │    │
//  │                        └─────────────┘          │  (MTKViewDelegate)   │
//  │                                                 └─────────────────┘    │
//  │                                                                         │
//  └─────────────────────────────────────────────────────────────────────────┘
//
//  플랫폼별 구현:
//  ─────────────────────────────────────────────────────────────────────────
//  - macOS: NSViewRepresentable 프로토콜 사용
//  - iOS/iPadOS/tvOS/visionOS: UIViewRepresentable 프로토콜 사용
//
//  OS 버전별 렌더러 선택:
//  ─────────────────────────────────────────────────────────────────────────
//  - macOS 26.0+ / iOS 26.0+: Metal4Renderer (Metal 4 API)
//  - 이전 버전: LegacyMetalRenderer (기존 Metal API)
//

import SwiftUI
import MetalKit

// ════════════════════════════════════════════════════════════════════════════
// MARK: - macOS Implementation
// ════════════════════════════════════════════════════════════════════════════

#if os(macOS)

// ════════════════════════════════════════════════════════════════════════════
// MARK: - Interactive MTKView (macOS)
// ════════════════════════════════════════════════════════════════════════════

/// 마우스 드래그를 지원하는 커스텀 MTKView
///
/// 마우스 드래그 이벤트를 캡처하여 렌더러의 쿼터니언 회전을 업데이트합니다.
class InteractiveMTKView: MTKView {

    /// 드래그 시작 위치
    private var dragStartLocation: NSPoint = .zero

    /// 현재 드래그 중인지 여부
    private var isDragging = false

    /// Metal 4 렌더러 참조 (약한 참조로 순환 참조 방지)
    @available(macOS 26.0, *)
    weak var metal4Renderer: Metal4Renderer? {
        get { _metal4Renderer as? Metal4Renderer }
        set { _metal4Renderer = newValue }
    }
    private weak var _metal4Renderer: AnyObject?

    // MARK: - Mouse Events

    override func mouseDown(with event: NSEvent) {
        isDragging = true
        dragStartLocation = event.locationInWindow

        // 렌더러에 드래그 시작 알림
        if #available(macOS 26.0, *) {
            metal4Renderer?.beginDrag()
        }
    }

    override func mouseDragged(with event: NSEvent) {
        guard isDragging else { return }

        let currentLocation = event.locationInWindow
        let deltaX = Float(currentLocation.x - dragStartLocation.x)
        let deltaY = Float(currentLocation.y - dragStartLocation.y)

        // 렌더러의 회전 업데이트
        if #available(macOS 26.0, *) {
            metal4Renderer?.updateRotation(deltaX: deltaX, deltaY: deltaY)
        }
    }

    override func mouseUp(with event: NSEvent) {
        isDragging = false
    }

    // 마우스 이벤트를 받기 위해 필요
    override var acceptsFirstResponder: Bool { true }
}

/// macOS용 Metal 뷰
///
/// `NSViewRepresentable` 프로토콜을 구현하여 SwiftUI에서
/// macOS의 네이티브 뷰(NSView)를 사용할 수 있게 합니다.
///
/// ## NSViewRepresentable 생명주기
///
/// ```
/// 1. makeCoordinator()
///    └─▶ Coordinator 인스턴스 생성
///        (SwiftUI 뷰와 NSView 사이의 통신 담당)
///
/// 2. makeNSView(context:)
///    └─▶ MTKView 인스턴스 생성 및 초기 설정
///        렌더러 생성 및 delegate 연결
///
/// 3. updateNSView(_:context:)
///    └─▶ SwiftUI 상태 변경 시 호출
///        (현재는 사용하지 않음)
///
/// 4. 뷰가 제거될 때 Coordinator가 자동으로 정리됨
/// ```
///
/// ## Metal 4 지원
///
/// OS 버전을 확인하여 적절한 렌더러를 선택합니다:
/// - macOS 26.0+: `Metal4Renderer`
/// - 이전 버전: `LegacyMetalRenderer`
struct MetalView: NSViewRepresentable {

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - NSViewRepresentable Protocol
    // ════════════════════════════════════════════════════════════════════════

    /// MTKView 인스턴스 생성
    ///
    /// SwiftUI가 이 뷰를 화면에 표시할 때 호출됩니다.
    /// MTKView를 생성하고 적절한 렌더러를 연결합니다.
    ///
    /// ## 구현 상세
    ///
    /// ```
    /// 1. MTKView 인스턴스 생성
    /// 2. OS 버전 확인
    /// 3. 적절한 렌더러 생성 (Metal4Renderer 또는 LegacyMetalRenderer)
    /// 4. 렌더러를 Coordinator에 저장 (메모리 관리)
    /// 5. MTKView의 delegate를 렌더러로 설정
    /// 6. MTKView 반환
    /// ```
    ///
    /// - Parameter context: SwiftUI가 제공하는 컨텍스트 (Coordinator 포함)
    /// - Returns: 설정된 MTKView 인스턴스
    func makeNSView(context: Context) -> MTKView {
        // ────────────────────────────────────────────────────────────────────
        // InteractiveMTKView 생성 (마우스 드래그 지원)
        // ────────────────────────────────────────────────────────────────────
        let mtkView = InteractiveMTKView()

        // ────────────────────────────────────────────────────────────────────
        // OS 버전에 따른 렌더러 선택
        // ────────────────────────────────────────────────────────────────────
        //
        // #available을 사용하여 컴파일 타임에 API 가용성을 확인합니다.
        // 런타임에 실제 OS 버전에 따라 적절한 브랜치가 실행됩니다.
        //
        // macOS 26.0 이상: Metal 4 API 사용 가능
        // macOS 26.0 미만: 기존 Metal API 사용
        //
        if #available(macOS 26.0, *) {
            // Metal 4 렌더러 생성 시도
            if let renderer = Metal4Renderer(mtkView: mtkView) {
                // Coordinator에 렌더러 저장 (강한 참조 유지)
                context.coordinator.metal4Renderer = renderer
                // InteractiveMTKView에 렌더러 참조 설정 (마우스 이벤트 전달용)
                mtkView.metal4Renderer = renderer
                // MTKView의 delegate를 렌더러로 설정
                // → 매 프레임 draw(in:) 메서드가 호출됨
                mtkView.delegate = renderer
            }
        } else {
            // 레거시 렌더러 생성 (하위 호환성)
            if let renderer = LegacyMetalRenderer(mtkView: mtkView) {
                context.coordinator.legacyRenderer = renderer
                mtkView.delegate = renderer
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // MTKView 설정
        // ────────────────────────────────────────────────────────────────────
        //
        // preferredFramesPerSecond: 목표 프레임 레이트
        //   - 60: 일반적인 디스플레이 (60Hz)
        //   - 120: ProMotion 디스플레이 지원 시
        //
        // enableSetNeedsDisplay: false
        //   - true: 명시적으로 setNeedsDisplay() 호출 시에만 다시 그림
        //   - false: preferredFramesPerSecond에 따라 자동으로 다시 그림
        //
        // isPaused: false
        //   - true: 렌더링 일시 정지
        //   - false: 렌더링 활성화
        //
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false

        return mtkView
    }

    /// NSView 업데이트
    ///
    /// SwiftUI의 상태가 변경되어 뷰를 업데이트해야 할 때 호출됩니다.
    /// 현재 구현에서는 외부 상태에 의존하지 않으므로 비어 있습니다.
    ///
    /// ## 사용 예시 (필요시)
    ///
    /// ```swift
    /// func updateNSView(_ nsView: MTKView, context: Context) {
    ///     // 예: 외부에서 회전 속도를 변경할 수 있는 경우
    ///     if #available(macOS 26.0, *) {
    ///         context.coordinator.metal4Renderer?.rotationSpeed = self.speed
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - nsView: 업데이트할 MTKView
    ///   - context: SwiftUI 컨텍스트
    func updateNSView(_ nsView: MTKView, context: Context) {
        // 현재는 업데이트할 외부 상태가 없음
    }

    /// Coordinator 생성
    ///
    /// SwiftUI와 NSView 사이의 통신을 담당하는 Coordinator를 생성합니다.
    /// Coordinator는 렌더러에 대한 강한 참조를 유지하여
    /// 렌더러가 메모리에서 해제되지 않도록 합니다.
    ///
    /// - Returns: 새로운 Coordinator 인스턴스
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Coordinator
    // ════════════════════════════════════════════════════════════════════════

    /// SwiftUI-NSView 브릿지 코디네이터
    ///
    /// 렌더러 인스턴스를 보관하고 SwiftUI와 MTKView 간의
    /// 통신을 중재하는 역할을 합니다.
    ///
    /// ## 역할
    ///
    /// 1. **렌더러 생명주기 관리**
    ///    - 렌더러에 대한 강한 참조 유지
    ///    - SwiftUI 뷰가 사라질 때 자동으로 정리됨
    ///
    /// 2. **타입 안전한 렌더러 접근**
    ///    - `@available` 속성을 사용하여 Metal 4 렌더러에 안전하게 접근
    ///    - 내부적으로 `Any` 타입으로 저장하여 컴파일러 오류 방지
    ///
    /// ## 메모리 관리
    ///
    /// ```
    /// SwiftUI View
    ///      │
    ///      ▼ (소유)
    /// Coordinator ─────▶ Renderer
    ///      │              (강한 참조)
    ///      │
    /// MTKView.delegate ───┘
    ///      (약한 참조처럼 동작)
    /// ```
    class Coordinator {
        /// Metal 4 렌더러 (macOS 26.0+)
        ///
        /// `@available` 속성을 사용하여 이 프로퍼티가
        /// macOS 26.0 이상에서만 Metal4Renderer 타입으로 접근 가능함을 표시합니다.
        ///
        /// 내부적으로는 `Any` 타입(`_metal4Renderer`)으로 저장하여
        /// 이전 OS 버전에서도 컴파일이 가능하도록 합니다.
        @available(macOS 26.0, *)
        var metal4Renderer: Metal4Renderer? {
            get { _metal4Renderer as? Metal4Renderer }
            set { _metal4Renderer = newValue }
        }

        /// Metal 4 렌더러의 실제 저장소 (타입 소거)
        ///
        /// `Any?` 타입을 사용하여 OS 버전과 관계없이 컴파일 가능하게 합니다.
        /// 실제 타입 캐스팅은 `metal4Renderer` 프로퍼티에서 수행됩니다.
        private var _metal4Renderer: Any?

        /// 레거시 Metal 렌더러 (이전 OS 버전용)
        ///
        /// macOS 26.0 미만에서 사용되는 기존 Metal API 기반 렌더러입니다.
        var legacyRenderer: LegacyMetalRenderer?
    }
}

#else

// ════════════════════════════════════════════════════════════════════════════
// MARK: - iOS/iPadOS/tvOS/visionOS Implementation
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// MARK: - Interactive MTKView (iOS)
// ════════════════════════════════════════════════════════════════════════════

/// 터치 드래그 및 핀치 줌을 지원하는 커스텀 MTKView
///
/// - 터치 드래그: 쿼터니언 기반 회전
/// - 핀치 줌: 카메라 거리 조절
class InteractiveMTKView: MTKView {

    /// 드래그 시작 위치
    private var dragStartLocation: CGPoint = .zero

    /// 핀치 줌 시작 시 두 터치 사이의 거리
    private var initialPinchDistance: CGFloat = 0

    /// Metal 4 렌더러 참조 (약한 참조로 순환 참조 방지)
    @available(iOS 26.0, *)
    weak var metal4Renderer: Metal4Renderer? {
        get { _metal4Renderer as? Metal4Renderer }
        set { _metal4Renderer = newValue }
    }
    private weak var _metal4Renderer: AnyObject?

    // MARK: - Helper Methods

    /// 두 터치 포인트 사이의 거리 계산
    private func distanceBetween(_ touch1: UITouch, _ touch2: UITouch) -> CGFloat {
        let point1 = touch1.location(in: self)
        let point2 = touch2.location(in: self)
        let dx = point2.x - point1.x
        let dy = point2.y - point1.y
        return sqrt(dx * dx + dy * dy)
    }

    // MARK: - Touch Events

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let allTouches = event?.allTouches else { return }

        if allTouches.count == 1 {
            // 단일 터치: 드래그 회전 시작
            guard let touch = touches.first else { return }
            dragStartLocation = touch.location(in: self)

            if #available(iOS 26.0, *) {
                metal4Renderer?.beginDrag()
            }
        } else if allTouches.count == 2 {
            // 두 손가락 터치: 핀치 줌 시작
            let touchArray = Array(allTouches)
            initialPinchDistance = distanceBetween(touchArray[0], touchArray[1])
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let allTouches = event?.allTouches else { return }

        if allTouches.count == 1 {
            // 단일 터치: 드래그 회전
            guard let touch = allTouches.first else { return }

            let currentLocation = touch.location(in: self)
            let deltaX = Float(currentLocation.x - dragStartLocation.x)
            let deltaY = Float(currentLocation.y - dragStartLocation.y)

            // iOS는 Y축이 아래로 증가하므로 deltaY 부호 반전
            if #available(iOS 26.0, *) {
                metal4Renderer?.updateRotation(deltaX: deltaX, deltaY: -deltaY)
            }
        } else if allTouches.count == 2 {
            // 두 손가락 터치: 핀치 줌
            let touchArray = Array(allTouches)
            let currentDistance = distanceBetween(touchArray[0], touchArray[1])

            if initialPinchDistance > 0 {
                let scale = Float(currentDistance / initialPinchDistance)
                if #available(iOS 26.0, *) {
                    metal4Renderer?.updateZoom(scale: scale)
                }
                // 현재 거리를 새로운 기준으로 설정 (점진적 줌)
                initialPinchDistance = currentDistance
            }
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let allTouches = event?.allTouches else { return }

        // 남은 터치가 1개이면 드래그 시작으로 전환
        if allTouches.count == 2 && touches.count == 1 {
            // 하나의 손가락만 떼면, 남은 손가락으로 드래그 시작
            let remainingTouches = allTouches.subtracting(touches)
            if let remainingTouch = remainingTouches.first {
                dragStartLocation = remainingTouch.location(in: self)
                if #available(iOS 26.0, *) {
                    metal4Renderer?.beginDrag()
                }
            }
        }

        initialPinchDistance = 0
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        initialPinchDistance = 0
    }
}

/// iOS용 Metal 뷰
///
/// `UIViewRepresentable` 프로토콜을 구현하여 SwiftUI에서
/// iOS의 네이티브 뷰(UIView)를 사용할 수 있게 합니다.
///
/// ## UIViewRepresentable 생명주기
///
/// ```
/// 1. makeCoordinator()
///    └─▶ Coordinator 인스턴스 생성
///
/// 2. makeUIView(context:)
///    └─▶ MTKView 인스턴스 생성 및 초기 설정
///        렌더러 생성 및 delegate 연결
///
/// 3. updateUIView(_:context:)
///    └─▶ SwiftUI 상태 변경 시 호출
///
/// 4. 뷰가 제거될 때 Coordinator가 자동으로 정리됨
/// ```
///
/// ## 플랫폼 지원
///
/// - iOS
/// - iPadOS
/// - tvOS
/// - visionOS
///
/// 모든 플랫폼에서 동일한 코드가 사용되며,
/// iOS 26.0 이상에서 Metal 4 API를 사용합니다.
struct MetalView: UIViewRepresentable {

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - UIViewRepresentable Protocol
    // ════════════════════════════════════════════════════════════════════════

    /// MTKView 인스턴스 생성 (iOS)
    ///
    /// SwiftUI가 이 뷰를 화면에 표시할 때 호출됩니다.
    /// MTKView를 생성하고 적절한 렌더러를 연결합니다.
    ///
    /// - Parameter context: SwiftUI가 제공하는 컨텍스트
    /// - Returns: 설정된 MTKView 인스턴스
    func makeUIView(context: Context) -> MTKView {
        // ────────────────────────────────────────────────────────────────────
        // InteractiveMTKView 생성 (터치 드래그 지원)
        // ────────────────────────────────────────────────────────────────────
        let mtkView = InteractiveMTKView()

        // ────────────────────────────────────────────────────────────────────
        // OS 버전에 따른 렌더러 선택
        // ────────────────────────────────────────────────────────────────────
        if #available(iOS 26.0, *) {
            // iOS 26.0+: Metal 4 렌더러
            if let renderer = Metal4Renderer(mtkView: mtkView) {
                // Coordinator에 렌더러 저장 (강한 참조 유지)
                context.coordinator.metal4Renderer = renderer
                // InteractiveMTKView에 렌더러 참조 설정 (터치 이벤트 전달용)
                mtkView.metal4Renderer = renderer
                // MTKView의 delegate를 렌더러로 설정
                mtkView.delegate = renderer
            }
        } else {
            // iOS 26.0 미만: 레거시 렌더러
            if let renderer = LegacyMetalRenderer(mtkView: mtkView) {
                context.coordinator.legacyRenderer = renderer
                mtkView.delegate = renderer
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // MTKView 설정
        // ────────────────────────────────────────────────────────────────────
        //
        // iOS에서의 추가 고려사항:
        // - ProMotion 디스플레이 (120Hz): preferredFramesPerSecond = 120
        // - 배터리 절약: 필요시 preferredFramesPerSecond 낮추기
        // - 백그라운드 진입 시: isPaused = true 권장
        //
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false

        // 멀티 터치 활성화 (핀치 줌용)
        mtkView.isMultipleTouchEnabled = true

        return mtkView
    }

    /// UIView 업데이트
    ///
    /// SwiftUI의 상태가 변경되어 뷰를 업데이트해야 할 때 호출됩니다.
    ///
    /// - Parameters:
    ///   - uiView: 업데이트할 MTKView
    ///   - context: SwiftUI 컨텍스트
    func updateUIView(_ uiView: MTKView, context: Context) {
        // 현재는 업데이트할 외부 상태가 없음
    }

    /// Coordinator 생성
    ///
    /// - Returns: 새로운 Coordinator 인스턴스
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    // ════════════════════════════════════════════════════════════════════════
    // MARK: - Coordinator
    // ════════════════════════════════════════════════════════════════════════

    /// SwiftUI-UIView 브릿지 코디네이터 (iOS)
    ///
    /// macOS 버전과 동일한 역할을 수행하며,
    /// `@available(iOS 26.0, *)` 속성을 사용합니다.
    class Coordinator {
        /// Metal 4 렌더러 (iOS 26.0+)
        @available(iOS 26.0, *)
        var metal4Renderer: Metal4Renderer? {
            get { _metal4Renderer as? Metal4Renderer }
            set { _metal4Renderer = newValue }
        }

        /// Metal 4 렌더러의 실제 저장소
        private var _metal4Renderer: Any?

        /// 레거시 Metal 렌더러 (이전 OS 버전용)
        var legacyRenderer: LegacyMetalRenderer?
    }
}

#endif
