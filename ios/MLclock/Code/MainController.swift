import UIKit
import PlanetSwift
import CoreML
import Vision

class MainController: PlanetViewController, CameraCaptureHelperDelegate {
    
    var captureHelper = CameraCaptureHelper(cameraPosition: .back)
    var lastOriginalImage:CIImage? = nil
    
    func playCameraImage(_ cameraCaptureHelper: CameraCaptureHelper, image: CIImage, originalImage: CIImage, frameNumber:Int, fps:Int) {
        
        if cameraCaptureHelper.perspectiveImagesCoords.count == 0 {
            
            // we want a sqaure area off of the top
            let w = originalImage.extent.width
            let h = originalImage.extent.height
            
            cameraCaptureHelper.perspectiveImagesCoords = [
                "inputTopLeft":CIVector(x:0, y: w),
                "inputTopRight":CIVector(x:w, y: w),
                "inputBottomLeft":CIVector(x:0, y: 0),
                "inputBottomRight":CIVector(x:w, y: 0),
                ]
        }
        
        DispatchQueue.main.async {
            self.preview.imageView.image = UIImage(ciImage: image)
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Main"
        mainBundlePath = "bundle://Assets/main/main.xml"
        loadView()
        
        captureHelper.delegate = self
        captureHelper.scaledImagesSize = CGSize(width: 128, height: 128)
        captureHelper.delegateWantsPerspectiveImages = true
        captureHelper.delegateWantsPlayImages = true
        
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

