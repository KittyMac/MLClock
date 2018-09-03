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
                    
                    
                    /*
                    // TODO: we need an algorithm to identify multiple hotspots as our model can identify
                    // multiple clock faces in one go.  Here's one idea:
                    
                    let output = results![0].featureValue.multiArrayValue!
                    let subdiv = output.count / 2

                    // 1. Create a grid which matches the grid of probabilities in dimensions
                    // 2. Find the highest probability above a certain threshold, do a flood fill algorithm off of that an mark off the spots
                    // 3. Repeat #2, but do not count previously used spots
                    
                    
                    // 0. average the X and Y probabilities into one probability map
                    var avgValues = Array(repeating: Array<Double>(repeating: 0.0, count: subdiv), count: subdiv)
                    for x in 0..<subdiv {
                        for y in 0..<subdiv {
                            avgValues[x][y] = (output[x].doubleValue * output[subdiv+y].doubleValue)
                        }
                    }
                    
                    var grid = Array(repeating: Array(repeating: 0, count: subdiv), count: subdiv)
                    let cropThreshold = 0.1
                    let peakThreshold = 0.99
                    var maxCropSize = 0
                    while true {
                        
                        // find the highest point above threshold
                        var peakValue = 0.0
                        var peakX = 0
                        var peakY = 0
                        for x in 0..<subdiv {
                            for y in 0..<subdiv {
                                if grid[x][y] == 0 && avgValues[x][y] >= peakValue {
                                    peakValue = avgValues[x][y]
                                    peakX = x
                                    peakY = y
                                }
                            }
                        }
                        
                        // if we don't find any value above the thresold, we're done
                        if peakValue < peakThreshold {
                            break
                        }
                        
                        // perform ez flood fill
                        var ezScan: ((Int,Int,Int) -> Void)!
                        var minX:Int = subdiv
                        var minY:Int = subdiv
                        var maxX:Int = 0
                        var maxY:Int = 0
                        var boxSize:Int = 0
                        
                        ezScan = { (startX, startY, direction) in
                            var x = startX
                            var y = startY
                            
                            while (x >= 0 && x < subdiv && y >= 0 && y < subdiv && grid[x][y] == 0 && avgValues[x][y] > cropThreshold) {
                                
                                grid[x][y] = 1
                                boxSize += 1
                                
                                if x < minX {
                                    minX = x
                                }
                                if y < minY {
                                    minY = y
                                }
                                if x > maxX {
                                    maxX = x
                                }
                                if y > maxY {
                                    maxY = x
                                }
                                
                                if direction == 0 { // scan to the right
                                    x += 1
                                } else if direction == 1 { // scan to the left
                                    x -= 1
                                } else if direction == 2 { // scan down
                                    y -= 1
                                } else if direction == 3 { // scan up
                                    y += 1
                                }
                            }
                        }
                        
                        ezScan(peakX, peakY, 0)
                        ezScan(peakX, peakY, 1)
                        ezScan(peakX, peakY, 2)
                        ezScan(peakX, peakY, 3)
                        
                        if boxSize > maxCropSize {
                            maxCropSize = boxSize
                            
                            print("peak: \(peakX),\(peakY) :: \(peakValue) :: \(boxSize)")
                            
                            let xscale = (newImage.extent.width / CGFloat(subdiv))
                            let yscale = (newImage.extent.height / CGFloat(subdiv))
                            
                            self.bestCropRect = CGRect(x: CGFloat(minX)*xscale, y: newImage.extent.height - CGFloat(maxY)*yscale, width: CGFloat(maxX-minX)*xscale, height: CGFloat(maxY-minY)*yscale)
                        }
                    }
                    
                    print("crop: \(self.bestCropRect)")
                    */
                    

                    
                    let output = results![0].featureValue.multiArrayValue!
                    let subdiv = output.count / 2
                    let xdelta = 1.0 / Double(subdiv)
                    let ydelta = 1.0 / Double(subdiv)
                    
                    var xmin = 1.0
                    var xmax = 0.0
                    var ymin = 1.0
                    var ymax = 0.0
                    
                    var avgXValues = [Double](repeating: 0, count: subdiv)
                    var avgYValues = [Double](repeating: 0, count: subdiv)
                    
                    for i in 0..<subdiv {
                        avgXValues[i] = output[i].doubleValue
                        avgYValues[i] = output[subdiv+i].doubleValue
                    }
                    
                    
                    if false {
                        // average by neighbors to bring down individual spikes
                        avgXValues[0] = (output[0].doubleValue + output[1].doubleValue) / 2.0
                        avgXValues[subdiv-1] = (output[subdiv-1].doubleValue + output[subdiv-2].doubleValue) / 2.0
                        
                        avgYValues[0] = (output[subdiv+0].doubleValue + output[subdiv+1].doubleValue) / 2.0
                        avgYValues[subdiv-1] = (output[subdiv+subdiv-1].doubleValue + output[subdiv+subdiv-2].doubleValue) / 2.0
                        
                        for i in 1..<subdiv-1 {
                            avgXValues[i] = (output[i].doubleValue + output[i-1].doubleValue + output[i+1].doubleValue) / 3.0
                            avgYValues[i] = (output[subdiv+i].doubleValue + output[subdiv+i-1].doubleValue + output[subdiv+i+1].doubleValue) / 3.0
                        }
                    }
                    
                    for x in 0..<subdiv {
                        for y in 0..<subdiv {
                            let xValue = (Double(x) * xdelta)
                            let yValue = (Double(y) * ydelta)
                            
                            if avgXValues[x] >= 0.99 && avgYValues[y] >= 0.99 {
                                if xValue < xmin {
                                    xmin = xValue
                                }
                                if xValue > xmax {
                                    xmax = xValue
                                }

                                if yValue < ymin {
                                    ymin = yValue
                                }
                                if yValue > ymax {
                                    ymax = yValue
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

