//
//  OBJLoader.swift
//  Metal4Render
//
//  OBJ 및 MTL 파일을 파싱하여 Metal에서 렌더링할 수 있는 메시 데이터로 변환
//

import Foundation
import simd

// MARK: - 데이터 구조체

/// 머티리얼 타입 열거형
enum MaterialType: Float {
    case standard = 0.0   // 기본 표면
    case glass = 1.0      // 유리 (투명, 반사)
    case metal = 2.0      // 금속 (높은 반사)
    case led = 3.0        // LED/발광체
    case rubber = 4.0     // 고무 (바퀴)
}

/// MTL 파일에서 파싱한 머티리얼 정보
struct OBJMaterial {
    var name: String
    var diffuseColor: SIMD4<Float>  // Kd 값 (RGB) + Alpha
    var ambientColor: SIMD4<Float>? // Ka 값 (Kd가 없을 때 fallback)
    var hasExplicitColor: Bool = false  // Kd 또는 Ka가 명시적으로 설정되었는지

    // PBR 머티리얼 속성
    var metallic: Float = 0.0      // 금속성 (0.0 ~ 1.0)
    var roughness: Float = 0.5     // 거칠기 (0.0 ~ 1.0)
    var emission: Float = 0.0      // 발광 강도
    var materialType: MaterialType = .standard

    init(name: String, diffuseColor: SIMD3<Float> = SIMD3<Float>(0.8, 0.8, 0.8)) {
        self.name = name
        self.diffuseColor = SIMD4<Float>(diffuseColor.x, diffuseColor.y, diffuseColor.z, 1.0)
    }

    /// 최종 색상 반환 (Kd > Ka > 기본색 순서로 우선)
    var finalColor: SIMD4<Float> {
        if hasExplicitColor {
            return diffuseColor
        } else if let ambient = ambientColor {
            return ambient
        }
        return diffuseColor
    }

    /// 머티리얼 파라미터를 SIMD4로 반환
    var materialParams: SIMD4<Float> {
        return SIMD4<Float>(metallic, roughness, emission, materialType.rawValue)
    }
}

/// OBJ 파일에서 로드된 메시 데이터
struct OBJMesh {
    /// 버텍스 배열 (position + color)
    var vertices: [Vertex]
    /// 인덱스 배열 (UInt32 - 65535개 이상의 버텍스 지원)
    var indices: [UInt32]
    /// 버텍스 수
    var vertexCount: Int { vertices.count }
    /// 인덱스 수
    var indexCount: Int { indices.count }
}

// MARK: - OBJLoader

/// OBJ 및 MTL 파일 로더
///
/// ## 사용 예시
/// ```swift
/// let loader = OBJLoader()
/// let mesh = try loader.load(
///     objURL: Bundle.main.url(forResource: "model", withExtension: "obj")!,
///     mtlURL: Bundle.main.url(forResource: "model", withExtension: "mtl")!
/// )
/// ```
class OBJLoader {

    // MARK: - 에러 정의

    enum LoaderError: Error, LocalizedError {
        case fileNotFound(String)
        case parseError(String)
        case invalidFaceFormat(String)

        var errorDescription: String? {
            switch self {
            case .fileNotFound(let path):
                return "파일을 찾을 수 없습니다: \(path)"
            case .parseError(let message):
                return "파싱 오류: \(message)"
            case .invalidFaceFormat(let line):
                return "잘못된 face 형식: \(line)"
            }
        }
    }

    // MARK: - Material Info Structure

    /// 버스 머티리얼 정보 구조체
    struct BusMaterialInfo {
        var color: SIMD4<Float>
        var metallic: Float
        var roughness: Float
        var emission: Float
        var materialType: MaterialType

        /// 머티리얼 파라미터를 SIMD4로 반환
        var materialParams: SIMD4<Float> {
            return SIMD4<Float>(metallic, roughness, emission, materialType.rawValue)
        }
    }

    // MARK: - Color Generation

    /// 버스 모델의 머티리얼 이름에 대한 사전 정의된 속성
    private static let busMaterials: [String: BusMaterialInfo] = [
        // 버스 차체 - 진한 남색/검정 (도장된 금속, 약간의 광택)
        "citybus3_dark": BusMaterialInfo(
            color: SIMD4<Float>(0.08, 0.10, 0.15, 1.0),
            metallic: 0.1,
            roughness: 0.35,
            emission: 0.0,
            materialType: .standard
        ),
        // LED 조명 - 밝은 노란색 (발광)
        "citybus_led": BusMaterialInfo(
            color: SIMD4<Float>(1.0, 0.85, 0.2, 1.0),
            metallic: 0.0,
            roughness: 0.3,
            emission: 2.0,  // 강한 발광
            materialType: .led
        ),
        // 거울 - 반사되는 크롬 (높은 금속성)
        "mirror_citybus_dark": BusMaterialInfo(
            color: SIMD4<Float>(0.9, 0.9, 0.92, 1.0),
            metallic: 0.95,
            roughness: 0.05,  // 매우 매끄러움
            emission: 0.0,
            materialType: .metal
        ),
        // 유리 - 투명한 파란색 (반투명, 반사)
        "glass": BusMaterialInfo(
            color: SIMD4<Float>(0.15, 0.25, 0.35, 0.4),
            metallic: 0.0,
            roughness: 0.05,  // 매우 매끄러움
            emission: 0.0,
            materialType: .glass
        ),
        "Glass_Clear": BusMaterialInfo(
            color: SIMD4<Float>(0.2, 0.3, 0.4, 0.35),
            metallic: 0.0,
            roughness: 0.02,
            emission: 0.0,
            materialType: .glass
        ),
        // 바퀴 - 검정 고무 (무광)
        "wheel": BusMaterialInfo(
            color: SIMD4<Float>(0.02, 0.02, 0.02, 1.0),
            metallic: 0.0,
            roughness: 0.9,  // 거친 고무
            emission: 0.0,
            materialType: .rubber
        ),
        // 파란색 버스 차체
        "Bus_Blue": BusMaterialInfo(
            color: SIMD4<Float>(0.05, 0.15, 0.4, 1.0),
            metallic: 0.15,
            roughness: 0.3,
            emission: 0.0,
            materialType: .standard
        ),
        // 헤드라이트
        "headlight": BusMaterialInfo(
            color: SIMD4<Float>(1.0, 1.0, 0.95, 1.0),
            metallic: 0.0,
            roughness: 0.1,
            emission: 3.0,
            materialType: .led
        ),
        // 테일라이트 (빨간색)
        "taillight": BusMaterialInfo(
            color: SIMD4<Float>(1.0, 0.1, 0.05, 1.0),
            metallic: 0.0,
            roughness: 0.1,
            emission: 2.5,
            materialType: .led
        ),
        // 크롬 범퍼/트림
        "chrome": BusMaterialInfo(
            color: SIMD4<Float>(0.95, 0.95, 0.97, 1.0),
            metallic: 1.0,
            roughness: 0.02,
            emission: 0.0,
            materialType: .metal
        )
    ]

    /// 머티리얼 이름으로 버스 머티리얼 정보 찾기
    /// - Parameter name: 머티리얼 이름
    /// - Returns: 매칭되는 머티리얼 정보 또는 nil
    private static func findBusMaterial(_ name: String) -> BusMaterialInfo? {
        // 정확한 매칭 시도
        if let material = busMaterials[name] {
            return material
        }
        // 부분 문자열 매칭 (예: "citybus3_dark.007" -> "citybus3_dark")
        for (key, material) in busMaterials {
            if name.lowercased().contains(key.lowercased()) {
                return material
            }
        }
        return nil
    }

    /// 머티리얼 이름으로 버스 색상 찾기 (하위 호환성)
    /// - Parameter name: 머티리얼 이름
    /// - Returns: 매칭되는 색상 또는 nil
    private static func findBusColor(_ name: String) -> SIMD4<Float>? {
        return findBusMaterial(name)?.color
    }

    /// 머티리얼 이름에서 머티리얼 정보 생성
    /// - Parameter name: 머티리얼 이름
    /// - Returns: 사전 정의 머티리얼 또는 이름 해시 기반의 기본 머티리얼
    private static func generateMaterialFromName(_ name: String) -> BusMaterialInfo {
        // 먼저 버스 관련 머티리얼 확인
        if let busMaterial = findBusMaterial(name) {
            return busMaterial
        }

        // 이름 해시를 사용하여 일관된 색상 생성
        var hash: UInt32 = 5381
        for char in name.utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt32(char)
        }

        // 해시를 HSV로 변환하여 선명한 색상 생성
        let hue = Float(hash % 360) / 360.0
        let saturation: Float = 0.7
        let value: Float = 0.9

        // HSV to RGB 변환
        let c = value * saturation
        let x = c * (1 - abs(fmod(hue * 6, 2) - 1))
        let m = value - c

        var r: Float, g: Float, b: Float
        let hueSegment = Int(hue * 6)

        switch hueSegment {
        case 0: (r, g, b) = (c, x, 0)
        case 1: (r, g, b) = (x, c, 0)
        case 2: (r, g, b) = (0, c, x)
        case 3: (r, g, b) = (0, x, c)
        case 4: (r, g, b) = (x, 0, c)
        default: (r, g, b) = (c, 0, x)
        }

        // 기본 머티리얼 속성 반환
        return BusMaterialInfo(
            color: SIMD4<Float>(r + m, g + m, b + m, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            emission: 0.0,
            materialType: .standard
        )
    }

    /// 머티리얼 이름에서 고유 색상 생성 (하위 호환성)
    /// - Parameter name: 머티리얼 이름
    /// - Returns: 사전 정의 색상 또는 이름 해시 기반의 RGBA 색상
    private static func generateColorFromName(_ name: String) -> SIMD4<Float> {
        return generateMaterialFromName(name).color
    }

    // MARK: - Public Methods

    /// OBJ 및 MTL 파일을 로드하여 메시 데이터 반환
    /// - Parameters:
    ///   - objURL: OBJ 파일 경로
    ///   - mtlURL: MTL 파일 경로 (옵션)
    /// - Returns: 파싱된 메시 데이터
    func load(objURL: URL, mtlURL: URL? = nil) throws -> OBJMesh {
        // 1. MTL 파일 파싱 (있는 경우)
        var materials: [String: OBJMaterial] = [:]
        if let mtlURL = mtlURL {
            materials = try parseMTL(url: mtlURL)
        }

        // 2. OBJ 파일 파싱
        return try parseOBJ(url: objURL, materials: materials)
    }

    // MARK: - MTL Parser

    /// MTL 파일을 파싱하여 머티리얼 딕셔너리 반환
    private func parseMTL(url: URL) throws -> [String: OBJMaterial] {
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            throw LoaderError.fileNotFound(url.path)
        }

        var materials: [String: OBJMaterial] = [:]
        var currentMaterial: OBJMaterial?

        let lines = content.components(separatedBy: .newlines)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // 빈 줄이나 주석 스킵
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }

            let parts = trimmed.split(separator: " ", omittingEmptySubsequences: true)
            guard let keyword = parts.first else { continue }

            switch keyword {
            case "newmtl":
                // 이전 머티리얼 저장
                if let material = currentMaterial {
                    materials[material.name] = material
                }
                // 새 머티리얼 시작
                let name = parts.dropFirst().joined(separator: " ")
                currentMaterial = OBJMaterial(name: name)

            case "Kd":
                // Diffuse color (RGB) - 우선순위 가장 높음
                if parts.count >= 4,
                   let r = Float(parts[1]),
                   let g = Float(parts[2]),
                   let b = Float(parts[3]) {
                    currentMaterial?.diffuseColor = SIMD4<Float>(r, g, b, 1.0)
                    currentMaterial?.hasExplicitColor = true
                }

            case "Ka":
                // Ambient color (RGB) - Kd가 없을 때 fallback으로 사용
                if parts.count >= 4,
                   let r = Float(parts[1]),
                   let g = Float(parts[2]),
                   let b = Float(parts[3]) {
                    currentMaterial?.ambientColor = SIMD4<Float>(r, g, b, 1.0)
                }

            case "d":
                // Dissolve (투명도)
                if parts.count >= 2,
                   let alpha = Float(parts[1]) {
                    currentMaterial?.diffuseColor.w = alpha
                }

            default:
                // 다른 속성은 무시 (Ns, Ks, Ke, Ni, illum, map_* 등)
                break
            }
        }

        // 마지막 머티리얼 저장
        if let material = currentMaterial {
            materials[material.name] = material
        }

        return materials
    }

    // MARK: - OBJ Parser

    /// OBJ 파일을 파싱하여 메시 데이터 반환
    private func parseOBJ(url: URL, materials: [String: OBJMaterial]) throws -> OBJMesh {
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            throw LoaderError.fileNotFound(url.path)
        }

        // 임시 저장소
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var texCoords: [SIMD2<Float>] = []

        // 최종 결과
        var vertices: [Vertex] = []
        var indices: [UInt32] = []

        // 버텍스 중복 제거를 위한 캐시
        // Key: "posIdx/texIdx/normalIdx", Value: 버텍스 인덱스
        var vertexCache: [String: UInt32] = [:]

        // 현재 머티리얼 정보
        var currentMaterialInfo = BusMaterialInfo(
            color: SIMD4<Float>(0.8, 0.8, 0.8, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            emission: 0.0,
            materialType: .standard
        )

        let lines = content.components(separatedBy: .newlines)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // 빈 줄이나 주석 스킵
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }

            let parts = trimmed.split(separator: " ", omittingEmptySubsequences: true)
            guard let keyword = parts.first else { continue }

            switch keyword {
            case "v":
                // 버텍스 위치
                if parts.count >= 4,
                   let x = Float(parts[1]),
                   let y = Float(parts[2]),
                   let z = Float(parts[3]) {
                    positions.append(SIMD3<Float>(x, y, z))
                }

            case "vn":
                // 버텍스 노말
                if parts.count >= 4,
                   let x = Float(parts[1]),
                   let y = Float(parts[2]),
                   let z = Float(parts[3]) {
                    normals.append(SIMD3<Float>(x, y, z))
                }

            case "vt":
                // 텍스처 좌표
                if parts.count >= 3,
                   let u = Float(parts[1]),
                   let v = Float(parts[2]) {
                    texCoords.append(SIMD2<Float>(u, v))
                }

            case "usemtl":
                // 머티리얼 변경
                let materialName = parts.dropFirst().joined(separator: " ")

                // 먼저 버스 머티리얼 정보 확인 (우선순위 높음)
                if let busMaterial = Self.findBusMaterial(materialName) {
                    currentMaterialInfo = busMaterial
                } else if let material = materials[materialName] {
                    // MTL 파일에서 로드된 머티리얼 사용
                    var matInfo = BusMaterialInfo(
                        color: material.finalColor,
                        metallic: material.metallic,
                        roughness: material.roughness,
                        emission: material.emission,
                        materialType: material.materialType
                    )

                    // Ka가 흰색(1,1,1)이고 Kd가 없으면 머티리얼 이름 기반 정보 생성
                    if !material.hasExplicitColor {
                        let isWhiteAmbient = material.ambientColor.map {
                            $0.x > 0.9 && $0.y > 0.9 && $0.z > 0.9
                        } ?? true
                        if isWhiteAmbient {
                            matInfo = Self.generateMaterialFromName(materialName)
                        }
                    }
                    currentMaterialInfo = matInfo
                } else {
                    // 머티리얼이 없으면 이름 기반 정보 생성
                    currentMaterialInfo = Self.generateMaterialFromName(materialName)
                }

            case "f":
                // Face (삼각형)
                // 형식: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 [v4/vt4/vn4]
                let faceVertices = Array(parts.dropFirst())

                // 삼각형으로 분할 (quad인 경우 두 개의 삼각형으로)
                // Triangle fan: 0-1-2, 0-2-3, 0-3-4, ...
                if faceVertices.count >= 3 {
                    var faceIndices: [UInt32] = []

                    for faceVertex in faceVertices {
                        let index = try processVertex(
                            faceVertex: String(faceVertex),
                            positions: positions,
                            texCoords: texCoords,
                            normals: normals,
                            materialInfo: currentMaterialInfo,
                            vertices: &vertices,
                            vertexCache: &vertexCache
                        )
                        faceIndices.append(index)
                    }

                    // Triangle fan으로 삼각형 생성
                    for i in 1..<(faceIndices.count - 1) {
                        indices.append(faceIndices[0])
                        indices.append(faceIndices[i])
                        indices.append(faceIndices[i + 1])
                    }
                }

            default:
                // 다른 키워드 무시 (o, g, s, mtllib 등)
                break
            }
        }

        return OBJMesh(vertices: vertices, indices: indices)
    }

    /// Face 버텍스를 처리하여 인덱스 반환
    /// - 형식: v, v/vt, v/vt/vn, v//vn
    private func processVertex(
        faceVertex: String,
        positions: [SIMD3<Float>],
        texCoords: [SIMD2<Float>],
        normals: [SIMD3<Float>],
        materialInfo: BusMaterialInfo,
        vertices: inout [Vertex],
        vertexCache: inout [String: UInt32]
    ) throws -> UInt32 {

        // 캐시 키 생성 (색상 및 머티리얼 포함)
        let cacheKey = "\(faceVertex)_\(materialInfo.color.x)_\(materialInfo.color.y)_\(materialInfo.color.z)_\(materialInfo.metallic)_\(materialInfo.roughness)"

        // 캐시에 있으면 기존 인덱스 반환
        if let cachedIndex = vertexCache[cacheKey] {
            return cachedIndex
        }

        // 인덱스 파싱
        // 형식: v, v/vt, v/vt/vn, v//vn
        let components = faceVertex.split(separator: "/", omittingEmptySubsequences: false)

        guard let posIndexStr = components.first,
              let posIndex = Int(posIndexStr),
              posIndex > 0,
              posIndex <= positions.count else {
            throw LoaderError.invalidFaceFormat(faceVertex)
        }

        // OBJ 인덱스는 1부터 시작
        let position = positions[posIndex - 1]

        // 노말 인덱스 파싱 (형식: v/vt/vn 또는 v//vn)
        var normal = SIMD3<Float>(0, 1, 0)  // 기본 노말 (위쪽)
        if components.count >= 3 {
            let normalIndexStr = components[2]
            if let normalIndex = Int(normalIndexStr),
               normalIndex > 0,
               normalIndex <= normals.count {
                normal = normals[normalIndex - 1]
            }
        }

        // 새 버텍스 생성 (노말 및 머티리얼 파라미터 포함)
        let vertex = Vertex(
            position: position,
            normal: normal,
            color: materialInfo.color,
            materialParams: materialInfo.materialParams
        )

        // 버텍스 배열에 추가
        let newIndex = UInt32(vertices.count)
        vertices.append(vertex)

        // 캐시에 저장
        vertexCache[cacheKey] = newIndex

        return newIndex
    }
}
