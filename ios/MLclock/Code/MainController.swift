import UIKit
import PlanetSwift
import CoreML
import Vision

class MainController: PlanetViewController, CameraCaptureHelperDelegate {
    
    var captureHelper = CameraCaptureHelper(cameraPosition: .back)
    var model:VNCoreMLModel? = nil
    var overrideImage:CIImage? = nil
    
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
        
        let handler = VNImageRequestHandler(ciImage: localImage)
        do {
            let request = VNCoreMLRequest(model: model!)
            try handler.perform([request])
            if let results = request.results as? [VNClassificationObservation] {
                // find the highest confidence hour
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
                
                if bestHour == 0 {
                    bestHour = 12
                }
                
                var clockString = String(format: "%02d:%02d", bestHour, bestMinute)
                
                if bestHourConfidence < 0.7 || bestMinuteConfidence < 0.7 {
                    clockString = "--:--"
                } else {
                    print("\(clockString) ---- \(bestHourConfidence)  \(bestMinuteConfidence)")
                }
                
                DispatchQueue.main.async {
                    self.clockLabel.label.text = clockString
                }
            }
            
        } catch {
            print(error)
        }
        
        DispatchQueue.main.async {
            self.preview.imageView.image = UIImage(ciImage: localImage)
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

    
    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Main"
        mainBundlePath = "bundle://Assets/main/main.xml"
        loadView()
        
        //overrideImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/main/debug/clock_02.46.png")))
        
        captureHelper.delegate = self
        captureHelper.scaledImagesSize = CGSize(width: 128, height: 128)
        captureHelper.delegateWantsScaledImages = true
        captureHelper.delegateWantsPerspectiveImages = true
        captureHelper.delegateWantsPlayImages = true
        
        loadModel()
        
        UIApplication.shared.isIdleTimerDisabled = true
    }
    
    fileprivate var clockLabel: Label {
        return mainXmlView!.elementForId("clockLabel")!.asLabel!
    }
    
    fileprivate var preview: ImageView {
        return mainXmlView!.elementForId("preview")!.asImageView!
    }
    
    fileprivate var scope: AnyObject? {
        return mainXmlView!.elementForId("root")!.asScene!.scope()
    }

}

