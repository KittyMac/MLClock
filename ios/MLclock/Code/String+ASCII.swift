import Foundation

extension String {
    
    subscript (bounds: CountableClosedRange<Int>) -> String {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return String(self[start...end])
    }
    
    subscript (bounds: CountableRange<Int>) -> String {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return String(self[start..<end])
    }
    
    var asciiArray8: [UInt8] {
        return unicodeScalars.filter{$0.isASCII}.map{UInt8($0.value)}
    }
    
    init (_ asciiArray8 : [UInt8]) {
        self.init()
        
        for x in asciiArray {
            self.append(Character(UnicodeScalar(UInt16(x))!))
        }
    }
    
    var asciiArray16: [UInt16] {
        return unicodeScalars.filter{$0.isASCII}.map{UInt16($0.value)}
    }
    
    init (_ asciiArray16 : [UInt16]) {
        self.init()
        
        for x in asciiArray {
            self.append(Character(UnicodeScalar(UInt16(x))!))
        }
    }
    
    
    var asciiArray: [UInt32] {
        return unicodeScalars.filter{$0.isASCII}.map{UInt32($0.value)}
    }
    
    init (_ asciiArray : [UInt32]) {
        self.init()
        
        for x in asciiArray {
            self.append(Character(UnicodeScalar(UInt32(x))!))
        }
    }
}
