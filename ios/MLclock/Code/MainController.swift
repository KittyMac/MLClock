import UIKit
import PlanetSwift
import CoreML
import Vision

class MainController: PlanetViewController, CameraCaptureHelperDelegate {
    
    var captureHelper = CameraCaptureHelper(cameraPosition: .back)
    var model:VNCoreMLModel? = nil
    var overrideImage:CIImage? = nil
    var displayedClickConfidence:Float = 0.0
    let ciContext = CIContext(options: [:])
    
    var objectLocalization = MLObjectLocalization()
    
    func playCameraImage(_ cameraCaptureHelper: CameraCaptureHelper, image: CIImage, originalImage: CIImage, frameNumber:Int, fps:Int) {
        
        var localImage = image
        
        if overrideImage != nil {
            localImage = overrideImage!
        }
        
        if cameraCaptureHelper.perspectiveImagesCoords.count == 0 {
            
            // we want a sqaure area off of the top
            let w = originalImage.extent.width
            
            cameraCaptureHelper.perspectiveImagesCoords = [
                "inputTopLeft":CIVector(x:0, y: w),
                "inputTopRight":CIVector(x:w, y: w),
                "inputBottomLeft":CIVector(x:0, y: 0),
                "inputBottomRight":CIVector(x:w, y: 0),
                ]
        }
        
        objectLocalization.updateImage(localImage)
        
        let bestCrop = objectLocalization.bestCrop()
        if bestCrop.size.width > 2.0 && bestCrop.size.height > 2.0 {
            detectTimeFromImage(localImage, objectLocalization.bestPerspective())
        }
        
        DispatchQueue.main.async {
            self.preview.imageView.image = UIImage(ciImage: localImage)
        }
    }
    
    
    func detectTimeFromImage(_ fullImage:CIImage, _ perspectiveImagesCoords:[String:Any]) {
        
        let extractedImage = fullImage.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords)
        
        DispatchQueue.main.async {
            self.cropPreview.imageView.image = UIImage(ciImage: extractedImage)
        }
        
        let handler = VNImageRequestHandler(ciImage: extractedImage)
        do {
            let request = VNCoreMLRequest(model: model!)
            try handler.perform([request])
            if let results = request.results as? [VNClassificationObservation] {
                // find the highest confidence hour
                var bestHour = 0
                var bestHourConfidence:Float = 0.0
                var notclockConfidence:Float = 0.0
                
                for result in results {
                    if result.identifier == "notclock" {
                        notclockConfidence = result.confidence
                        break
                    }
                }
                
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
                
                if bestHour == 0 {
                    bestHour = 12
                }
                                
                DispatchQueue.main.async {
                
                    if (notclockConfidence + notclockConfidence > self.displayedClickConfidence * 2.0) {
                        self.displayedClickConfidence = notclockConfidence
                        self.clockLabel.label.text = "no clock"
                        self.clockConfidence.label.text = String(format: "%0.2f%%", (bestHourConfidence + bestMinuteConfidence) * 50.0)
                    }
                    
                    if (bestHourConfidence + bestMinuteConfidence > self.displayedClickConfidence) {
                        self.displayedClickConfidence = bestHourConfidence + bestMinuteConfidence
                        self.clockLabel.label.text = String(format: "%02d:%02d", bestHour, bestMinute)
                        self.clockConfidence.label.text = String(format: "%0.2f%%", (bestHourConfidence + bestMinuteConfidence) * 50.0)
                    }
                }
                
                displayedClickConfidence -= 0.01
            }
            
        } catch {
            print(error)
        }
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

    
    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Main"
        mainBundlePath = "bundle://Assets/main/main.xml"
        loadView()
        
        //overrideImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/main/debug/full_clock8.jpg")))
        //overrideImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/main/debug/localization_test.png")))
        
        captureHelper.delegate = self
        captureHelper.delegateWantsPerspectiveImages = true
        captureHelper.delegateWantsPlayImages = true
        captureHelper.maxDesiredImageResolution = 1280 * 720
        captureHelper.delegateWantsSquareCrop = true
        
        loadModel()
        
        UIApplication.shared.isIdleTimerDisabled = true
    }
    
    fileprivate var clockLabel: Label {
        return mainXmlView!.elementForId("clockLabel")!.asLabel!
    }
    
    fileprivate var clockConfidence: Label {
        return mainXmlView!.elementForId("clockConfidence")!.asLabel!
    }
    
    fileprivate var preview: ImageView {
        return mainXmlView!.elementForId("preview")!.asImageView!
    }
    
    fileprivate var cropPreview: ImageView {
        return mainXmlView!.elementForId("cropPreview")!.asImageView!
    }
    
    fileprivate var scope: AnyObject? {
        return mainXmlView!.elementForId("root")!.asScene!.scope()
    }

}

