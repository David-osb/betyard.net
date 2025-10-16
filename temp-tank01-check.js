// Check if Tank01 functions are being called
console.log('=== TANK01 FUNCTION CHECK ===');

// Override Tank01 functions to prevent any calls
window.fetchNFLDataWithTank01Enhanced = function() {
    console.log('🚫 fetchNFLDataWithTank01Enhanced: BLOCKED BY OVERRIDE');
    return Promise.resolve(false);
};

window.fetchNFLDataWithTank01 = function() {
    console.log('🚫 fetchNFLDataWithTank01: BLOCKED BY OVERRIDE');
    return Promise.resolve(false);
};

console.log('✅ Tank01 functions overridden and blocked');