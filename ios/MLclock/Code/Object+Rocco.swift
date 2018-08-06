// extenstions to planetswift to make it more usable for me.

import Foundation
import PlanetSwift

extension Object {
    
    func new<T:Button>(_ styleId:String?, _ block:((T) -> Void)) -> T {
        // the most common use case when creating a new entity programmatically is to assign its styles (if any), then
        // override those styles (programmatically), then call gaxb prepare.
        // this method takes care of all of that
        
        self.styleId = styleId
        if let styleElement = Object.styleForId(self.styleId!) {
            _ = styleElement.imprintAttributes(self)
        }
        
        block(self as! T)
        
        self.gaxbPrepare()
        
        return self as! T
    }
    
    func new<T:Button>(block:((T) -> Void)) -> T {
        return new(nil, block)
    }
    
}
