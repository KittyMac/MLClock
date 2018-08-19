import UIKit
import PlanetSwift
import CoreML
import Vision

class MLObjectLocalization {
    
    private var bestCropRect:CGRect = CGRect.zero
    
    let ciContext = CIContext(options: [:])
    
    private var handler = VNSequenceRequestHandler()
    var model:VNCoreMLModel? = nil
        
    func updateImage(_ newImage:CIImage) {
        
        loadModel()
        
        if self.model != nil {
            do {
                let request = VNCoreMLRequest(model: self.model!)
                try self.handler.perform([request], on: newImage)
                
                let results = request.results as? [VNClassificationObservation]
                if results != nil {
                    var xmin:CGFloat = 0
                    var ymin:CGFloat = 0
                    var xmax:CGFloat = 0
                    var ymax:CGFloat = 0
                    
                    for result in results! {
                        if result.identifier == "xmin" {
                            xmin = CGFloat(result.confidence)
                        }
                        if result.identifier == "ymin" {
                            ymin = CGFloat(result.confidence)
                        }
                        if result.identifier == "xmax" {
                            xmax = CGFloat(result.confidence)
                        }
                        if result.identifier == "ymax" {
                            ymax = CGFloat(result.confidence)
                        }
                    }
                    
                    // stupid: need to convert from model coords to real image coords
                    let xscale = newImage.extent.width
                    let yscale = newImage.extent.height
                    
                    self.bestCropRect = CGRect(x: xmin*xscale, y: yscale - ymax*yscale, width: (xmax-xmin)*xscale, height: (ymax-ymin)*yscale)
                }
                
            } catch {
                print(error)
            }
        }
    }
    
    func bestCrop() -> CGRect {
        return bestCropRect
    }
    
    func bestPerspective() -> [String:Any] {
        let perspectiveImageCoords = [
            "inputTopLeft":CIVector(x: bestCropRect.minX, y: bestCropRect.maxY),
            "inputTopRight":CIVector(x: bestCropRect.maxX, y: bestCropRect.maxY),
            "inputBottomLeft":CIVector(x: bestCropRect.minX, y: bestCropRect.minY),
            "inputBottomRight":CIVector(x: bestCropRect.maxX, y: bestCropRect.minY)
        ]
        return perspectiveImageCoords

    }
    
    func loadModel() {
        if self.model == nil {
            do {
                let modelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/main/clock.mlmodel"))
                let compiledUrl = try MLModel.compileModel(at: modelURL)
                let model = try MLModel(contentsOf: compiledUrl)
                self.model = try? VNCoreMLModel(for: model)
            } catch {
                print(error)
            }
        }
    }
    
}

