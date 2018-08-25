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
                
                let results = request.results as? [VNCoreMLFeatureValueObservation]
                if results != nil {
                    let output = results![0].featureValue.multiArrayValue!
                    let subdiv = output.count / 2
                    let xdelta = 1.0 / Double(subdiv)
                    let ydelta = 1.0 / Double(subdiv)
                    
                    var xmin = 1.0
                    var xmax = 0.0
                    var ymin = 1.0
                    var ymax = 0.0
                    
                    var avgXValues = [Int](repeating: 0, count: subdiv)
                    var avgYValues = [Int](repeating: 0, count: subdiv)
                    
                    for i in 0..<subdiv {
                        avgXValues[i] = Int(output[i].doubleValue * 100)
                        avgYValues[i] = Int(output[subdiv+i].doubleValue * 100)
                    }
                    
                    /*
                    if true {
                        // average by neighbors to bring down individual spikes
                        avgXValues[0] = (output[0].doubleValue + output[1].doubleValue) / 2.0
                        avgXValues[subdiv-1] = (output[subdiv-1].doubleValue + output[subdiv-2].doubleValue) / 2.0
                        
                        avgYValues[0] = (output[subdiv+0].doubleValue + output[subdiv+1].doubleValue) / 2.0
                        avgYValues[subdiv-1] = (output[subdiv+subdiv-1].doubleValue + output[subdiv+subdiv-2].doubleValue) / 2.0
                        
                        for i in 1..<subdiv-1 {
                            avgXValues[i] = (output[i].doubleValue + output[i-1].doubleValue + output[i+1].doubleValue) / 3.0
                            avgYValues[i] = (output[subdiv+i].doubleValue + output[subdiv+i-1].doubleValue + output[subdiv+i+1].doubleValue) / 3.0
                        }
                    }*/
                    
                    for x in 0..<subdiv {
                        for y in 0..<subdiv {
                            let xValue = (Double(x) * xdelta)
                            let yValue = (Double(y) * ydelta)
                            
                            if avgXValues[x] >= 20 && avgYValues[y] >= 20 {
                                if xValue < xmin {
                                    xmin = xValue
                                }
                                if xValue + xdelta > xmax {
                                    xmax = xValue + xdelta
                                }

                                if yValue < ymin {
                                    ymin = yValue
                                }
                                if yValue + ydelta > ymax {
                                    ymax = yValue + ydelta
                                }
                            }
                        }
                    }
                    
                    let xscale = newImage.extent.width
                    let yscale = newImage.extent.height
                    
                    self.bestCropRect = CGRect(x: CGFloat(xmin)*xscale, y: yscale - CGFloat(ymax)*yscale, width: CGFloat(xmax-xmin)*xscale, height: CGFloat(ymax-ymin)*yscale)
                    
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

