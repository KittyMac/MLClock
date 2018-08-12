import UIKit
import PlanetSwift
import CoreML
import Vision

class MLObjectLocalization {
    
    private let modelInputSize = 8
    private var currentImage:CIImage? = nil
    private var bestCropRect:CGRect = CGRect.zero
    
    let ciContext = CIContext(options: [:])
    var calibrationImage:CIImage? = nil
    
    private var handler = VNSequenceRequestHandler()
    
    var model:VNCoreMLModel? = nil
    var shouldContinueRunningGA = true
    
    class Organism {
        // Note: our organism contains the data we need to calcuate the
        // four points of our crop. It is ideal for us to use the least
        // amount of data to do this (so the GA has less to figure out).
        // As such, we are going to use the following mechanism
        //
        // content[0] == x == center x position normalized [0.0, 1.0]
        // content[1] == y == center y position normalized [0.0, 1.0]
        // content[3] == radius == half size of the square crop normalized  [0.0, 0.25]
        // content[4] == skew == value to allow skew (TBD)
        
        let contentLength = 3
        
        var x:CGFloat = 0.5
        var y:CGFloat = 0.5
        var radius:CGFloat = 0.1
        
        subscript(index:Int) -> CGFloat {
            get {
                switch index {
                case 0: return x
                case 1: return y
                default: return radius
                }
            }
            set(v) {
                switch index {
                case 0: x = v
                case 1: y = v
                default: radius = v
                }
            }
        }
        
        func validate() {
            // crop must be completely contained.
            if radius < 0.01 {
                radius = 0.01
            }
            if x < radius {
               x = radius
            }
            if x >= 0.99999 - radius {
                x = 0.99999 - radius
            }
            if y < radius {
                y = radius
            }
            if y >= 0.99999 - radius {
                y = 0.99999 - radius
            }
        }
        
        func randomize(_ index:Int, _ prng:PRNG) {
            switch index {
            case 0: x = prng.getRandomNumberCGf()
            case 1: y = prng.getRandomNumberCGf()
            default: radius = prng.getRandomNumberCGf() * 0.5
            }
        }
        
        func randomizeAll(_ prng:PRNG) {
            x = prng.getRandomNumberCGf()
            y = prng.getRandomNumberCGf()
            radius = prng.getRandomNumberCGf()
        }
        
        func randomizeSome(_ prng:PRNG) {
            if prng.getRandomNumberf() < 0.5 {
                x = prng.getRandomNumberCGf()
            }
            if prng.getRandomNumberf() < 0.5 {
                y = prng.getRandomNumberCGf()
            }
            if prng.getRandomNumberf() < 0.5 {
                radius = prng.getRandomNumberCGf()
            }
        }
        
        func randomizeAdjust(_ prng:PRNG) {
            if prng.getRandomNumberf() < 0.5 {
                x += prng.getRandomNumberCGf() * 0.2 - 0.1
            }
            if prng.getRandomNumberf() < 0.5 {
                y += prng.getRandomNumberCGf() * 0.2 - 0.1
            }
            if prng.getRandomNumberf() < 0.5 {
                radius += prng.getRandomNumberCGf() * 0.2 - 0.1
            }
        }
        
        func normalizedCrop() -> CGRect {
            return CGRect(x: x-radius, y: y-radius, width: radius*2.0, height: radius*2.0)
        }
        
        func fullsizeCrop(_ w:CGFloat, _ h:CGFloat) -> CGRect {
            return CGRect(x: (x-radius) * h,
                          y: (y-radius) * w,
                          width: (radius*2.0) * h,
                          height: (radius*2.0) * h)
        }
    }
    
    func updateImage(_ newImage:CIImage) {
        let cgimg = self.ciContext.createCGImage(newImage, from: newImage.extent)
        let imgData = UIImageJPEGRepresentation(UIImage(cgImage: cgimg!), 1.0)
        currentImage = CIImage(data: imgData!)
    }
    
    func runGA() {
        
        guard let model = model else {
            return
        }
        
        if currentImage == nil {
            return
        }
        
        // Load our calibration image and convert to RGB bytes
        calibrationImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/main/calibration.png")))
        let cgImage = self.ciContext.createCGImage(calibrationImage!, from: calibrationImage!.extent)
        var calibrationRGBBytes = [UInt8](repeating: 0, count: 1)
        
        if cgImage != nil {
            let width = cgImage!.width
            let height = cgImage!.height
            let bitsPerComponent = cgImage!.bitsPerComponent
            let rowBytes = width * 1
            let totalBytes = height * width * 1
            
            calibrationRGBBytes = [UInt8](repeating: 0, count: totalBytes)
            
            let colorSpace = CGColorSpaceCreateDeviceGray()
            let contextRef = CGContext(data: &calibrationRGBBytes, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: rowBytes, space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue)
            
            contextRef?.draw(cgImage!, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
            
            autolevels(width, height, &calibrationRGBBytes)
        }
        
        let w = currentImage!.extent.width
        let h = currentImage!.extent.height
        
        let ga = GeneticAlgorithm<Organism>()
        
        ga.numberOfOrganisms = 200
        
        ga.generateOrganism = { (idx, prng) in
            let newChild = Organism ()
            newChild.randomizeAll(prng)
            newChild.validate()
            return newChild;
        }
        
        ga.breedOrganisms = { (organismA, organismB, child, prng) in
            if organismB == nil {
                // create a new random member of the population
                child.randomizeAll(prng)
                return
            }
            
            if (organismA === organismB) {
                for i in 0..<child.contentLength {
                    child [i] = organismA [i]
                }
                if prng.getRandomNumberf() < 0.5 {
                    child.randomizeSome(prng)
                } else {
                    child.randomizeAdjust(prng)
                }
            } else {
                for i in 0..<child.contentLength {
                    let t = prng.getRandomNumberf()
                    if (t < 0.5) {
                        child [i] = organismA [i];
                    } else {
                        child [i] = organismB! [i];
                    }
                }
                
                child.randomizeSome(prng)
            }
            child.validate()
        }
        
        ga.scoreOrganism = { (organism, threadIdx, prng) in
            
            return autoreleasepool { () -> Float in
                
                let fullsizeCrop = organism.fullsizeCrop(w, h)
                if fullsizeCrop.maxX >= w || fullsizeCrop.maxY >= h {
                    return 0.0
                }
                
                let perspectiveImagesCoords = [
                    "inputTopLeft":CIVector(x:fullsizeCrop.minX, y: fullsizeCrop.maxY),
                    "inputTopRight":CIVector(x:fullsizeCrop.maxX, y: fullsizeCrop.maxY),
                    "inputBottomLeft":CIVector(x:fullsizeCrop.minX, y: fullsizeCrop.minY),
                    "inputBottomRight":CIVector(x:fullsizeCrop.maxX, y: fullsizeCrop.minY)
                    ]
                
                // 1. use coreimage to extract it
                let extractedImage = self.currentImage!.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords)

                
                
                let targetSize:CGFloat = self.calibrationImage!.extent.width
                let adjustedImage = extractedImage.transformed(by: CGAffineTransform(scaleX: targetSize / extractedImage.extent.width, y: targetSize / extractedImage.extent.height))
                
                guard let cgImage = self.ciContext.createCGImage(adjustedImage, from: adjustedImage.extent) else {
                    return 0.0
                }

                let width = cgImage.width
                let height = cgImage.height
                let bitsPerComponent = cgImage.bitsPerComponent
                let rowBytes = width * 1
                let totalBytes = height * width * 1
                
                var rgbBytes = [UInt8](repeating: 0, count: totalBytes)
                
                let colorSpace = CGColorSpaceCreateDeviceGray()
                let contextRef = CGContext(data: &rgbBytes, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: rowBytes, space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue)
                contextRef?.draw(cgImage, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
                
                
                // run over both images, and determine how different they are...
                if calibrationRGBBytes.count == rgbBytes.count {
                    var totalDiff:Double = 0.0
                    
                    self.autolevels(width, height, &rgbBytes)
                    
                    for i in 0..<(width*height) {
                        totalDiff = totalDiff + abs(Double(calibrationRGBBytes[i]) - Double(rgbBytes[i]))
                    }
                    return 1.0 - Float(totalDiff / Double(width*height*255))
                }
                
                /*
                do {
                    let request = VNCoreMLRequest(model: model)
                
                    try self.handler.perform([request], on: extractedImage)
                    
                    guard let results = request.results as? [VNClassificationObservation] else {
                        return 0.0
                    }
                    var confidence:Float = 0.0
                    for result in results {
                        if result.identifier == "clock" {
                            confidence = result.confidence
                        }
                    }
                    return confidence
                } catch {
                    print(error)
                }
                */
                
                return 0.0
            }
        }
        
        ga.chosenOrganism = { (organism, score, generation, sharedOrganismIdx, prng) in
            print("...\(score)")
            
            self.bestCropRect = organism.fullsizeCrop(w, h)

            print("score: \(score)\n    x:\(organism.x)\n    y:\(organism.y)\n    radius:\(organism.radius)")
            
            if score >= 1.0 {
                return true
            }
            return false
        }
        
        
        let timeout = -1
        //let timeout = 10000
        let bestOrganism = ga.PerformGenetics (Int64(timeout))
        let bestAccuracy = ga.scoreOrganism(bestOrganism, 1, PRNG())
        print("score: \(bestAccuracy)\n    x:\(bestOrganism.x)\n    y:\(bestOrganism.y)\n    radius:\(bestOrganism.radius)")
    }
    
    func bestCrop() -> CGRect {
        return bestCropRect
    }
    
    func autolevels(_ width:Int, _ height:Int, _ bytes:inout [UInt8]) {
        var min:CGFloat = 255
        var max:CGFloat = 0
        for i in 0..<(width*height) {
            let grey = CGFloat(bytes[i])
            if grey > max {
                max = grey
            }
            if grey < min {
                min = grey
            }
        }
        
        for i in 0..<(width*height) {
            bytes[i] = UInt8((CGFloat(bytes[i]) - min) * (255.0 / (max - min)))
        }
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
        
        shouldContinueRunningGA = true
        DispatchQueue.global(qos: .userInteractive).async {
            while self.shouldContinueRunningGA {
                self.runGA()
            }
        }
    }
    
    func end() {
        shouldContinueRunningGA = false
    }

}

