// Constants (from constants.py)
const A_PATAH = '\u05b7';      // a
const E_TSERE = '\u05b5';      // e
const I_HIRIK = '\u05b4';      // i
const O_HOLAM = '\u05b9';      // o
const U_QUBUT = '\u05bb';      // u
const U_DAGESH = '\u05bc';     // u/dagesh mark
const S_SIN = '\u05c2';        // s (sin)
const STRESS_HATAMA = '\u05ab'; // stress mark

const CAN_HAVE_DAGESH = 'בכפו';
const CAN_HAVE_SIN = 'ש';
const LETTERS = 'אבגדהוזחטיכלמנסעפצקרשת' + 'םןףץ';

// Deduplicate map (from deduplicate.py)
const DEDUPLICATE_MAP = {
    // Row 1
    '\u0590': '', '\u0591': '', '\u0592': '', '\u0593': '',
    '\u0594': '', '\u0595': '', '\u0596': '', '\u0597': '',
    '\u0598': '', '\u0599': '', '\u059a': '', '\u059b': '',
    '\u059c': '', '\u059d': '', '\u059e': '', '\u059f': '',
    // Row 2
    '\u05a0': '', '\u05a1': '', '\u05a2': '', '\u05a3': '',
    '\u05a4': '', '\u05a5': '', '\u05a6': '', '\u05a7': '',
    '\u05a8': '', '\u05a9': '', '\u05aa': '',
    '\u05ab': '\u05ab', // Hatama
    '\u05ac': '', '\u05ad': '', '\u05ae': '', '\u05af': '',
    // Row 3
    '\u05b0': '',        // Shva -> nothing
    '\u05b1': '\u05b5',  // Hataf segol -> Tsere (e)
    '\u05b2': '\u05b7',  // Hataf patah -> Patah (a)
    '\u05b3': '\u05b9',  // Hataf qamats -> Holam (o)
    '\u05b4': '\u05b4',  // Hiriq -> Hirik (i)
    '\u05b5': '\u05b5',  // Tsere -> Tsere (e)
    '\u05b6': '\u05b5',  // Segol -> Tsere (e)
    '\u05b7': '\u05b7',  // Patah -> Patah (a)
    '\u05b8': '\u05b7',  // Qamats -> Patah (a)
    '\u05b9': '\u05b9',  // Holam -> Holam (o)
    '\u05ba': '\u05b9',  // Holam haser for Vav -> Holam (o)
    '\u05bb': '\u05bb',  // Qubut -> Qubut (u)
    '\u05bc': '\u05bc',  // Dagesh -> Dagesh (u/dagesh mark)
    '\u05bd': '',        // Meteg -> nothing
    '\u05be': '',        // Maqaf -> nothing
    '\u05bf': '',        // Rafe -> nothing
    // Row 4
    '\u05c0': '',        // Paseq -> nothing
    '\u05c1': '',        // Shin dot -> nothing
    '\u05c2': '\u05c2',  // Sin -> Sin (s)
    '\u05c3': '',        // Sof pasuq -> nothing
    '\u05c4': '',        // Upper dot -> nothing
    '\u05c5': '',        // Lower dot -> nothing
    '\u05c6': '',        // Nun Hafukha -> nothing
    '\u05c7': '\u05b9',  // Qamats qatan -> Holam (o)
    '\u05c8': '', '\u05c9': '', '\u05ca': '', '\u05cb': '',
    '\u05cc': '', '\u05cd': '', '\u05ce': '', '\u05cf': '',
    // Row 5
    '\u05d0': '\u05d0', // Alef
    '\u05d1': '\u05d1', // Bet
    '\u05d2': '\u05d2', // Gimel
    '\u05d3': '\u05d3', // Dalet
    '\u05d4': '\u05d4', // Hey
    '\u05d5': '\u05d5', // Vav
    '\u05d6': '\u05d6', // Zayin
    '\u05d7': '\u05d7', // Het
    '\u05d8': '\u05d8', // Tet
    '\u05d9': '\u05d9', // Yod
    '\u05da': '\u05da', // Final kaf
    '\u05db': '\u05db', // Kaf
    '\u05dc': '\u05dc', // Lamed
    '\u05dd': '\u05dd', // Mem sofit
    '\u05de': '\u05de', // Mem
    '\u05df': '\u05df', // Nun sofit
    // Row 6
    '\u05e0': '\u05e0', // Nun
    '\u05e1': '\u05e1', // Samech
    '\u05e2': '\u05e2', // Ayin
    '\u05e3': '\u05e3', // Pe sofit
    '\u05e4': '\u05e4', // Pe
    '\u05e5': '\u05e5', // Tsadi sofit
    '\u05e6': '\u05e6', // Tsadi
    '\u05e7': '\u05e7', // Kuf
    '\u05e8': '\u05e8', // Resh
    '\u05e9': '\u05e9', // Shin
    '\u05ea': '\u05ea', // Taf
    '\u05eb': '\u05eb', '\u05ec': '\u05ec', '\u05ed': '\u05ed',
    '\u05ee': '\u05ee', '\u05ef': '\u05ef',
    // Row 7
    '\u05f0': '', '\u05f1': '', '\u05f2': '',
    '\u05f3': "'",      // Geresh -> Single quote
    '\u05f4': '', '\u05f5': '', '\u05f6': '', '\u05f7': '',
    '\u05f8': '', '\u05f9': '', '\u05fa': '', '\u05fb': '',
    '\u05fc': '', '\u05fd': '', '\u05fe': '', '\u05ff': '',
};

/**
 * Sort diacritics after each letter
 * @param {string} text
 * @returns {string}
 */
function sortDiacritics(text) {
    // Match a letter followed by one or more diacritics
    // \p{L} matches any letter (Unicode Letter category)
    // \p{M} matches any mark/diacritic (Unicode Mark category)
    const regex = /(\p{L})(\p{M}+)/gu;
    
    return text.replace(regex, (match, letter, diacritics) => {
        if (!diacritics) return letter;
        
        // Sort diacritics alphabetically
        const sortedDiacritics = diacritics.split('').sort().join('');
        return letter + sortedDiacritics;
    });
}

/**
 * Clean dagesh from letters that cannot have it
 * @param {string} text
 * @returns {string}
 */
function cleanDagesh(text) {
    const dagesh = '\u05bc';
    const canHaveDagesh = 'בכפו';
    
    // Match a letter followed by one or more diacritics
    // \p{L} matches any letter (Unicode Letter category)
    // \p{M} matches any mark/diacritic (Unicode Mark category)
    const regex = /(\p{L})(\p{M}+)/gu;
    
    return text.replace(regex, (match, letter, diacritics) => {
        if (!diacritics) return letter;
        
        // If letter cannot have dagesh, remove it
        if (!canHaveDagesh.includes(letter)) {
            const cleanedDiacritics = diacritics.split('')
                .filter(d => d !== dagesh)
                .join('');
            return letter + cleanedDiacritics;
        }
        
        return letter + diacritics;
    });
}

/**
 * Deduplicate characters using the map
 * @param {string} text
 * @returns {string}
 */
function deduplicate(text) {
    let result = '';
    for (let char of text) {
        if (char in DEDUPLICATE_MAP) {
            result += DEDUPLICATE_MAP[char];
        } else {
            result += char;
        }
    }
    return result;
}

/**
 * Main normalize function
 * @param {string} text
 * @returns {string}
 */
function normalize(text) {
    // NFD normalization (decompose)
    text = text.normalize('NFD');
    
    // Sort diacritics
    text = sortDiacritics(text);
    
    // Clean dagesh
    text = cleanDagesh(text);
    
    // Deduplicate
    text = deduplicate(text);
    
    return text;
}

// Setup textarea event listener
const textArea = document.getElementById('textArea');
let isNormalizing = false;

textArea.addEventListener('input', function(e) {
    if (isNormalizing) return;
    
    isNormalizing = true;
    
    // Save cursor position
    const start = this.selectionStart;
    const end = this.selectionEnd;
    
    // Get current value
    const originalText = this.value;
    
    // Normalize the text
    const normalizedText = normalize(originalText);
    
    // Only update if changed
    if (originalText !== normalizedText) {
        this.value = normalizedText;
        
        // Restore cursor position (approximately)
        // This is tricky because normalization may change text length
        const lengthDiff = normalizedText.length - originalText.length;
        const newStart = Math.max(0, start + lengthDiff);
        const newEnd = Math.max(0, end + lengthDiff);
        
        this.setSelectionRange(newStart, newEnd);
    }
    
    isNormalizing = false;
});

// Test with the example from main.py
const testText = "שִׁירוֹתִים בַּלִשכַּה וֵסִרטוֹן אֵחַד שֵהוּדלַף";
console.log('Original:', testText);
console.log('Normalized:', normalize(testText));

// Nikud panel functionality
const nikudButtons = document.querySelectorAll('.nikud-button');

nikudButtons.forEach(button => {
    button.addEventListener('click', function() {
        const nikud = this.getAttribute('data-nikud');
        const textArea = document.getElementById('textArea');
        
        // Get current cursor position
        const start = textArea.selectionStart;
        const end = textArea.selectionEnd;
        const text = textArea.value;
        
        // Insert nikud at cursor position
        const newText = text.slice(0, end) + nikud + text.slice(end);
        textArea.value = newText;
        
        // Move cursor after the inserted nikud
        textArea.selectionStart = textArea.selectionEnd = end + nikud.length;
        
        // Focus back on textarea
        textArea.focus();
        
        // Trigger normalization
        textArea.dispatchEvent(new Event('input'));
    });
});