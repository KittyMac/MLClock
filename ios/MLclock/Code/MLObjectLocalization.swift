import UIKit
import PlanetSwift
import CoreML
import Vision

class MLObjectLocalization {
    
    private let modelInputSize = 8
    private var currentImage:CIImage? = nil
    private var bestCropRect:CGRect = CGRect.zero
    private var bestPerspectiveCoords:[String:Any] = [:]
    private var bestScore:Float = 0.0
    
    let ciContext = CIContext(options: [:])
    var calibrationImage:CIImage? = nil
    
    var watchStart = DispatchTime.now()
    var lastGenerationCount = 0
    
    private var handler = VNSequenceRequestHandler()
    
    var model:VNCoreMLModel? = nil
    var shouldContinueRunningGA = true
    
    var calibrationRGBBytes = [UInt8](repeating: 0, count: 1)
    
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
        var skewX:CGFloat = 0.5
        var skewY:CGFloat = 0.5
        
        subscript(index:Int) -> CGFloat {
            get {
                switch index {
                case 0: return x
                case 1: return y
                case 2: return radius
                case 3: return skewX
                default: return skewY
                }
            }
            set(v) {
                switch index {
                case 0: x = v
                case 1: y = v
                case 2: radius = v
                case 3: skewX = v
                default: skewY = v
                }
            }
        }
        
        func validate() {
            // crop must be completely contained.
            if radius < 0.005 {
                radius = 0.005
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
            if skewX < 0 {
                skewX = 0
            }
            if skewX > 1 {
                skewX = 1
            }
            if skewY < 0 {
                skewY = 0
            }
            if skewY > 1 {
                skewY = 1
            }
        }
        
        func randomize(_ index:Int, _ prng:PRNG) {
            switch index {
            case 0: x = prng.getRandomNumberCGf()
            case 1: y = prng.getRandomNumberCGf()
            case 2: radius = prng.getRandomNumberCGf() * 0.5
            case 3: skewX = prng.getRandomNumberCGf()
            default: skewY = prng.getRandomNumberCGf()
            }
        }
        
        func randomizeAll(_ prng:PRNG) {
            x = prng.getRandomNumberCGf()
            y = prng.getRandomNumberCGf()
            radius = prng.getRandomNumberCGf()
            skewX = prng.getRandomNumberCGf()
            skewY = prng.getRandomNumberCGf()
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
            if prng.getRandomNumberf() < 0.5 {
                skewX = prng.getRandomNumberCGf()
            }
            if prng.getRandomNumberf() < 0.5 {
                skewY = prng.getRandomNumberCGf()
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
            if prng.getRandomNumberf() < 0.5 {
                skewX += prng.getRandomNumberCGf() * 0.2 - 0.1
            }
            if prng.getRandomNumberf() < 0.5 {
                skewY += prng.getRandomNumberCGf() * 0.2 - 0.1
            }
        }
        
        func fullsizeCrop(_ w:CGFloat, _ h:CGFloat) -> CGRect {
            return CGRect(x: (x-radius) * w,
                          y: (y-radius) * h,
                          width: (radius*2.0) * h,
                          height: (radius*2.0) * h)
        }
        
        public func lerp(_ min: CGFloat, _ max: CGFloat, _ t:CGFloat) -> CGFloat {
            return min + (t * (max - min))
        }
        
        func perspectiveCoords(_ w:CGFloat, _ h:CGFloat) -> [String:Any] {
            let minSkew:CGFloat = 0.9
            let maxSkew:CGFloat = 1.1
            let leftSkew = lerp(maxSkew, minSkew, skewX)
            let rightSkew = lerp(minSkew, maxSkew, skewX)
            let topSkew = lerp(maxSkew, minSkew, skewY)
            let bottomSkew = lerp(minSkew, maxSkew, skewY)
            
            let perspectiveImageCoords = [
                "inputTopLeft":CIVector(x: (x*w-radius*h*topSkew), y: (y*w+radius*h*leftSkew)),
                "inputTopRight":CIVector(x: (x*w+radius*h*topSkew), y: (y*w+radius*h*rightSkew)),
                "inputBottomLeft":CIVector(x: (x*w-radius*h*bottomSkew), y: (y*w-radius*h*leftSkew)),
                "inputBottomRight":CIVector(x: (x*w+radius*h*bottomSkew), y: (y*w-radius*h*rightSkew))
            ]
            return perspectiveImageCoords
        }
        
        func scoreCoreML(_ model:VNCoreMLModel, _ handler:VNSequenceRequestHandler, _ extractedImage:CIImage) -> Float {
            if extractedImage.extent.width < 2 || extractedImage.extent.height < 2 {
                return 0.0
            }
            
            do {
                let request = VNCoreMLRequest(model: model)
                
                try handler.perform([request], on: extractedImage)
                
                guard let results = request.results as? [VNClassificationObservation] else {
                    return 0.0
                }
                
                var notclockConfidence:Float = 0.0
                
                for result in results {
                    if result.identifier == "notclock" {
                        notclockConfidence = result.confidence
                        break
                    }
                }

                
                var bestHour = 0
                var bestHourConfidence:Float = 0.0
                
                for i in 0...12 {
                    let identifier = "hour\(i)"
                    for result in results {
                        if result.identifier == identifier {
                            let confidence = result.confidence
                            if confidence > bestHourConfidence {
                                bestHourConfidence = confidence
                                bestHour = i
                            }
                        }
                    }
                }
                
                // find the highest confidence minute
                var bestMinute = 0
                var bestMinuteConfidence:Float = 0.0
                
                for i in 0...60 {
                    let identifier = "minute\(i)"
                    for result in results {
                        if result.identifier == identifier {
                            let confidence = result.confidence
                            if confidence > bestMinuteConfidence {
                                bestMinuteConfidence = confidence
                                bestMinute = i
                            }
                        }
                    }
                }
                
                return (bestMinuteConfidence + bestHourConfidence) * 0.5 - notclockConfidence * 0.3
                
            } catch {
                print(error)
            }
            return 0.0
        }
        
        func scoreSimpleMatch(_ calibrationRGBBytes:[UInt8], _ calibrationImage:CIImage, _ extractedImage:CIImage, _ ciContext:CIContext) -> Float {
            let targetSize:CGFloat = calibrationImage.extent.width
            let adjustedImage = extractedImage.transformed(by: CGAffineTransform(scaleX: targetSize / extractedImage.extent.width, y: targetSize / extractedImage.extent.height))
            
            guard let cgImage = ciContext.createCGImage(adjustedImage, from: adjustedImage.extent) else {
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
                Organism.autolevels(width, height, &rgbBytes)
                for i in 0..<(width*height) {
                    totalDiff = totalDiff + abs(Double(calibrationRGBBytes[i]) - Double(rgbBytes[i]))
                }
                
                let score = 1.0 - Float(totalDiff / Double(width*height*255))
                return score
            }
            return 0.0
        }
        
        static func autolevels(_ width:Int, _ height:Int, _ bytes:inout [UInt8]) {
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
            
            if (max - min) <= 0 {
                return
            }
            
            for i in 0..<(width*height) {
                bytes[i] = UInt8((CGFloat(bytes[i]) - min) * (255.0 / (max - min)))
            }
        }
    }
    
    func workingImage() -> CIImage{
        return currentImage!
    }
    
    func updateImage(_ newImage:CIImage) {
        let cgimg = self.ciContext.createCGImage(newImage, from: newImage.extent)
        let imgData = UIImagePNGRepresentation(UIImage(cgImage: cgimg!))
        //let imgData = UIImageJPEGRepresentation(UIImage(cgImage: cgimg!), 1.0)
        currentImage = CIImage(data: imgData!)
    }
    
    func loadCalibrationImage() {
        // Load our calibration image and convert to RGB bytes
        calibrationImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/main/calibration_64.png")))
        let cgImage = ciContext.createCGImage(calibrationImage!, from: calibrationImage!.extent)
        
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
            
            Organism.autolevels(width, height, &calibrationRGBBytes)
        }
    }
    
    func runGA() {
        
        guard let model = model else {
            return
        }
        
        if currentImage == nil {
            return
        }
        
        
        
        let w = currentImage!.extent.width
        let h = currentImage!.extent.height
        
        let ga = GeneticAlgorithm<Organism>()
        
        ga.numberOfOrganisms = 200
        
        ga.adjustPopulation = { (population, populationScores, generationCount, prng) in
            // if the current best score is stuck low, throw away everything and start fresh
            let maxToClear = population.count-1
            if populationScores.last! > 0.8 {
                return
            }
            for idx in 0..<maxToClear {
                population[idx]!.randomizeAll(prng)
                population[idx]!.validate()
            }
            
            if self.calibrationImage != nil {
                
                print("begin grided population")
                
                // start some of the population out with a grid, to help spread out the initial search
                let temp = Organism()
                var gridScores:[Float] = [Float](repeating: 0.0, count: ga.numberOfOrganisms / 2)
                for s in [0.05, 0.1, 0.2, 0.3, 0.4] {

                    var x = s
                    while x < 1.0 - s {
                        
                        var y = s
                        while y < 1.0 - s {
                            
                            temp.x = CGFloat(x)
                            temp.y = CGFloat(y)
                            temp.radius = CGFloat(s)
                            
                            let perspectiveImagesCoords = temp.perspectiveCoords(w, h)
                            let extractedImage = self.currentImage!.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords)
                            let tempScore = temp.scoreSimpleMatch(self.calibrationRGBBytes, self.calibrationImage!, extractedImage, self.ciContext)
                            for i in 0..<gridScores.count {
                                if tempScore > gridScores[i] {
                                    gridScores[i] = tempScore
                                    population[i]?.x = temp.x
                                    population[i]?.y = temp.y
                                    population[i]?.radius = temp.radius
                                    population[i]?.skewX = temp.skewX
                                    population[i]?.skewY = temp.skewY
                                    population[i]?.validate()
                                    break
                                }
                            }
                            
                            y += s/2
                        }
                        
                        x += s/2
                    }
                }
                
                print("... end grided population")
            }
        }
        
        ga.generateOrganism = { (idx, prng) in
            let newChild = Organism ()
            newChild.randomizeAll(prng)
            newChild.validate()
            return newChild;
        }
        
        ga.breedOrganisms = { (organismA, organismB, child, prng) in
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
                        child [i] = organismB [i];
                    }
                }
                
                child.randomizeSome(prng)
            }
            child.validate()
        }
        
        ga.scoreOrganism = { (organism, threadIdx, prng) in
            
            let millisecondsElapsed = ((DispatchTime.now().uptimeNanoseconds - self.watchStart.uptimeNanoseconds) / 1000000)
            if millisecondsElapsed > 1000 {
                self.watchStart = DispatchTime.now()
                
                print("\(ga.numberOfGenerations - self.lastGenerationCount) generations/sec")
                self.lastGenerationCount = ga.numberOfGenerations
            }
                        
            return autoreleasepool { () -> Float in
                
                let fullsizeCrop = organism.fullsizeCrop(w, h)
                if fullsizeCrop.maxX >= w || fullsizeCrop.maxY >= h {
                    return 0.0
                }
                
                let perspectiveImagesCoords = organism.perspectiveCoords(w, h)
                let extractedImage = self.currentImage!.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords)

                /*
                let useCoreML = false
                if !useCoreML {
                    return organism.scoreSimpleMatch(self.calibrationRGBBytes, self.calibrationImage!, extractedImage, self.ciContext)
                } else {
                    return organism.scoreCoreML(self.model!, self.handler, extractedImage)
                }*/
                
                return organism.scoreSimpleMatch(self.calibrationRGBBytes, self.calibrationImage!, extractedImage, self.ciContext) + organism.scoreCoreML(self.model!, self.handler, extractedImage)
            }
        }
        
        ga.chosenOrganism = { (organism, score, generation, sharedOrganismIdx, prng) in
            self.bestCropRect = organism.fullsizeCrop(w, h)
            self.bestPerspectiveCoords = organism.perspectiveCoords(w, h)

            print("score: \(score)\n    x:\(organism.x)\n    y:\(organism.y)\n    radius:\(organism.radius)\n    skewX:\(organism.skewX)\n    skewY:\(organism.skewY)")
            
            if score >= 1.0 {
                return true
            }
            return false
        }
        
        
        let timeout = -1
        let bestOrganism = ga.PerformGenetics (Int64(timeout))
        let bestAccuracy = ga.scoreOrganism(bestOrganism, 1, PRNG())
        print("score: \(bestAccuracy)\n    x:\(bestOrganism.x)\n    y:\(bestOrganism.y)\n    radius:\(bestOrganism.radius)")
    }
    
    func bestCrop() -> CGRect {
        return bestCropRect
    }
    
    func bestConfidence() -> Float {
        return bestScore
    }
    
    func bestPerspective() -> [String:Any] {
        return bestPerspectiveCoords
    }
    
    func loadModel() {
        do {
            let modelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/main/time.mlmodel"))
            let compiledUrl = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiledUrl)
            self.model = try? VNCoreMLModel(for: model)
        } catch {
            print(error)
        }
    }
    
    func begin() {
        loadModel()
        loadCalibrationImage()
        
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

