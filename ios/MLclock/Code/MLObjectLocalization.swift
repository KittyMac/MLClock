import UIKit
import PlanetSwift
import CoreML
import Vision

class MLObjectLocalization {
    
    private var currentImage:CIImage? = nil
    private var bestCropRect:CGRect = CGRect.zero
    
    let ciContext = CIContext(options: [:])
    
    private var handler = VNSequenceRequestHandler()
    var model:VNCoreMLModel? = nil
    
    var shouldContinueRunning = false
    
    func workingImage() -> CIImage{
        return currentImage!
    }
    
    func updateImage(_ newImage:CIImage) {
        let cgimg = self.ciContext.createCGImage(newImage, from: newImage.extent)
        let imgData = UIImagePNGRepresentation(UIImage(cgImage: cgimg!))
        //let imgData = UIImageJPEGRepresentation(UIImage(cgImage: cgimg!), 1.0)
        currentImage = CIImage(data: imgData!)
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
        do {
            let modelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/main/clock.mlmodel"))
            let compiledUrl = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiledUrl)
            self.model = try? VNCoreMLModel(for: model)
        } catch {
            print(error)
        }
    }
    
    func begin() {
        loadModel()
        
        shouldContinueRunning = true
        DispatchQueue.global(qos: .userInteractive).async {
            while self.shouldContinueRunning {
                if self.model != nil && self.currentImage != nil {
                    do {
                        let request = VNCoreMLRequest(model: self.model!)
                        try self.handler.perform([request], on: self.currentImage!)
                        
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
                            let xscale = self.currentImage!.extent.width / 128.0
                            let yscale = self.currentImage!.extent.height / 128.0
                            
                            self.bestCropRect = CGRect(x: xmin*xscale, y: ymin*yscale, width: (xmax-xmin)*xscale, height: (ymax-ymin)*yscale)
                        }
                        
                    } catch {
                        print(error)
                    }
                }
            }
        }
    }
    
    func end() {
        shouldContinueRunning = false
    }

}

