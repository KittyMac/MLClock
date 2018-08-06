import UIKit
import PlanetSwift

extension UIView {
    
    func fillSuperview(_ top:Int, _ left:Int, _ bottom:Int, _ right:Int) {
        guard let constraintsView = findConstraintsView() else {
            return
        }
        
        constraintsView.addConstraint(NSLayoutConstraint(item: self, attribute: .top, relatedBy: .equal, toItem: superview, attribute: .top, multiplier: 1, constant: CGFloat(top)))
        constraintsView.addConstraint(NSLayoutConstraint(item: self, attribute: .left, relatedBy: .equal, toItem: superview, attribute: .left, multiplier: 1, constant: CGFloat(left)))
        constraintsView.addConstraint(NSLayoutConstraint(item: self, attribute: .bottom, relatedBy: .equal, toItem: superview, attribute: .bottom, multiplier: 1, constant: CGFloat(-bottom)))
        constraintsView.addConstraint(NSLayoutConstraint(item: self, attribute: .right, relatedBy: .equal, toItem: superview, attribute: .right, multiplier: 1, constant: CGFloat(-right)))
        
        self.translatesAutoresizingMaskIntoConstraints = false
        constraintsView.translatesAutoresizingMaskIntoConstraints = false
    }
    
    func fillSuperview() {
        guard let constraintsView = findConstraintsView() else {
            return
        }
        
        constraintsView.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .bottom))
        constraintsView.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .right))
        constraintsView.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .left))
        constraintsView.addConstraint(NSLayoutConstraint(item: self, toItem: superview, equalAttribute: .top))
        self.translatesAutoresizingMaskIntoConstraints = false
        constraintsView.translatesAutoresizingMaskIntoConstraints = false
    }
    
    func findConstraintsView() -> UIView? {
        return superview
    }
}
