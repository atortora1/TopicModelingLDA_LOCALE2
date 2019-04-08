import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import spacy
from spacy.lang.en import English
parser = English()
import math

#nltk.download('popular')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('wordnet')
spacy.load('en_core_web_sm')

def doubledecode(word,convertdate, as_unicode=True):
    try:
        if word is not None:
            # remove the windows gremlins O^1
            if '&aposs' in word:
                word = word.replace('&aposs', "")
            for src, dest in cp1252.items():
                word = word.replace(src, dest)
            word = word.encode('raw_unicode_escape')
            if as_unicode:
                # return as unicode string
                word = word.decode('utf8', 'ignore')
            if convertdate is True:
                word = word.replace('-','')
            else:
                word = word.strip().lower()
            if 'reuters.' in word:
                word = word.replace('reuters.', "")
            if 'bloomberg' in word:
                word = word.replace('bloomberg', "")
            if 'investorplace' in word:
                word = word.replace('investorplace', "")
            if 'frankfurt' in word:
                word = word.replace('frankfurt', "")
            if 'investing.com' in word:
                word = word.replace('investing.com', "")
            if 'reuters' in word:
                word = word.replace('reuters', "")
            if 'new york (reuters)' in word:
                word = word.replace('new york (reuters)', "")
            return word
    except Exception:
        print(word)

cp1252 = {
    u"\u0000": u"\x00",  # NULL
    u"\u0001": u"\x01",  # START OF HEADING
    u"\u0002": u"\x02",  # START OF TEXT
    u"\u0003": u"\x03",  # END OF TEXT
    u"\u0004": u"\x04",  # END OF TRANSMISSION
    u"\u0005": u"\x05",  # ENQUIRY
    u"\u0006": u"\x06",  # ACKNOWLEDGE
    u"\u0007": u"\x07",  # BELL
    u"\u0008": u"\x08",  # BACKSPACE
    u"\u0009": u"\x09",  # HORIZONTAL TABULATION
    u"\u000A": u"\x0A",  # LINE FEED
    u"\u000B": u"\x0B",  # VERTICAL TABULATION
    u"\u000C": u"\x0C",  # FORM FEED
    u"\u000D": u"\x0D",  # CARRIAGE RETURN
    u"\u000E": u"\x0E",  # SHIFT OUT
    u"\u000F": u"\x0F",  # SHIFT IN
    u"\u0010": u"\x10",  # DATA LINK ESCAPE
    u"\u0011": u"\x11",  # DEVICE CONTROL ONE
    u"\u0012": u"\x12",  # DEVICE CONTROL TWO
    u"\u0013": u"\x13",  # DEVICE CONTROL THREE
    u"\u0014": u"\x14",  # DEVICE CONTROL FOUR
    u"\u0015": u"\x15",  # NEGATIVE ACKNOWLEDGE
    u"\u0016": u"\x16",  # SYNCHRONOUS IDLE
    u"\u0017": u"\x17",  # END OF TRANSMISSION BLOCK
    u"\u0018": u"\x18",  # CANCEL
    u"\u0019": u"\x19",  # END OF MEDIUM
    u"\u001A": u"\x1A",  # SUBSTITUTE
    u"\u001B": u"\x1B",  # ESCAPE
    u"\u001C": u"\x1C",  # FILE SEPARATOR
    u"\u001D": u"\x1D",  # GROUP SEPARATOR
    u"\u001E": u"\x1E",  # RECORD SEPARATOR
    u"\u001F": u"\x1F",  # UNIT SEPARATOR
    u"\u0020": u"\x20",  # SPACE
    u"\u0021": u"\x21",  # EXCLAMATION MARK
    u"\u0022": u"\x22",  # QUOTATION MARK
    u"\u0023": u"\x23",  # NUMBER SIGN
    u"\u0024": u"\x24",  # DOLLAR SIGN
    u"\u0025": u"\x25",  # PERCENT SIGN
    u"\u0026": u"\x26",  # AMPERSAND
    u"\u0027": u"\x27",  # APOSTROPHE
    u"\u0028": u"\x28",  # LEFT PARENTHESIS
    u"\u0029": u"\x29",  # RIGHT PARENTHESIS
    u"\u002A": u"\x2A",  # ASTERISK
    u"\u002B": u"\x2B",  # PLUS SIGN
    u"\u002C": u"\x2C",  # COMMA
    u"\u002D": u"\x2D",  # HYPHEN-MINUS
    u"\u002E": u"\x2E",  # FULL STOP
    u"\u002F": u"\x2F",  # SOLIDUS
    u"\u0030": u"\x30",  # DIGIT ZERO
    u"\u0031": u"\x31",  # DIGIT ONE
    u"\u0032": u"\x32",  # DIGIT TWO
    u"\u0033": u"\x33",  # DIGIT THREE
    u"\u0034": u"\x34",  # DIGIT FOUR
    u"\u0035": u"\x35",  # DIGIT FIVE
    u"\u0036": u"\x36",  # DIGIT SIX
    u"\u0037": u"\x37",  # DIGIT SEVEN
    u"\u0038": u"\x38",  # DIGIT EIGHT
    u"\u0039": u"\x39",  # DIGIT NINE
    u"\u003A": u"\x3A",  # COLON
    u"\u003B": u"\x3B",  # SEMICOLON
    u"\u003C": u"\x3C",  # LESS-THAN SIGN
    u"\u003D": u"\x3D",  # EQUALS SIGN
    u"\u003E": u"\x3E",  # GREATER-THAN SIGN
    u"\u003F": u"\x3F",  # QUESTION MARK
    u"\u0040": u"\x40",  # COMMERCIAL AT
    u"\u0041": u"\x41",  # LATIN CAPITAL LETTER A
    u"\u0042": u"\x42",  # LATIN CAPITAL LETTER B
    u"\u0043": u"\x43",  # LATIN CAPITAL LETTER C
    u"\u0044": u"\x44",  # LATIN CAPITAL LETTER D
    u"\u0045": u"\x45",  # LATIN CAPITAL LETTER E
    u"\u0046": u"\x46",  # LATIN CAPITAL LETTER F
    u"\u0047": u"\x47",  # LATIN CAPITAL LETTER G
    u"\u0048": u"\x48",  # LATIN CAPITAL LETTER H
    u"\u0049": u"\x49",  # LATIN CAPITAL LETTER I
    u"\u004A": u"\x4A",  # LATIN CAPITAL LETTER J
    u"\u004B": u"\x4B",  # LATIN CAPITAL LETTER K
    u"\u004C": u"\x4C",  # LATIN CAPITAL LETTER L
    u"\u004D": u"\x4D",  # LATIN CAPITAL LETTER M
    u"\u004E": u"\x4E",  # LATIN CAPITAL LETTER N
    u"\u004F": u"\x4F",  # LATIN CAPITAL LETTER O
    u"\u0050": u"\x50",  # LATIN CAPITAL LETTER P
    u"\u0051": u"\x51",  # LATIN CAPITAL LETTER Q
    u"\u0052": u"\x52",  # LATIN CAPITAL LETTER R
    u"\u0053": u"\x53",  # LATIN CAPITAL LETTER S
    u"\u0054": u"\x54",  # LATIN CAPITAL LETTER T
    u"\u0055": u"\x55",  # LATIN CAPITAL LETTER U
    u"\u0056": u"\x56",  # LATIN CAPITAL LETTER V
    u"\u0057": u"\x57",  # LATIN CAPITAL LETTER W
    u"\u0058": u"\x58",  # LATIN CAPITAL LETTER X
    u"\u0059": u"\x59",  # LATIN CAPITAL LETTER Y
    u"\u005A": u"\x5A",  # LATIN CAPITAL LETTER Z
    u"\u005B": u"\x5B",  # LEFT SQUARE BRACKET
    u"\u005C": u"\x5C",  # REVERSE SOLIDUS
    u"\u005D": u"\x5D",  # RIGHT SQUARE BRACKET
    u"\u005E": u"\x5E",  # CIRCUMFLEX ACCENT
    u"\u005F": u"\x5F",  # LOW LINE
    u"\u0060": u"\x60",  # GRAVE ACCENT
    u"\u0061": u"\x61",  # LATIN SMALL LETTER A
    u"\u0062": u"\x62",  # LATIN SMALL LETTER B
    u"\u0063": u"\x63",  # LATIN SMALL LETTER C
    u"\u0064": u"\x64",  # LATIN SMALL LETTER D
    u"\u0065": u"\x65",  # LATIN SMALL LETTER E
    u"\u0066": u"\x66",  # LATIN SMALL LETTER F
    u"\u0067": u"\x67",  # LATIN SMALL LETTER G
    u"\u0068": u"\x68",  # LATIN SMALL LETTER H
    u"\u0069": u"\x69",  # LATIN SMALL LETTER I
    u"\u006A": u"\x6A",  # LATIN SMALL LETTER J
    u"\u006B": u"\x6B",  # LATIN SMALL LETTER K
    u"\u006C": u"\x6C",  # LATIN SMALL LETTER L
    u"\u006D": u"\x6D",  # LATIN SMALL LETTER M
    u"\u006E": u"\x6E",  # LATIN SMALL LETTER N
    u"\u006F": u"\x6F",  # LATIN SMALL LETTER O
    u"\u0070": u"\x70",  # LATIN SMALL LETTER P
    u"\u0071": u"\x71",  # LATIN SMALL LETTER Q
    u"\u0072": u"\x72",  # LATIN SMALL LETTER R
    u"\u0073": u"\x73",  # LATIN SMALL LETTER S
    u"\u0074": u"\x74",  # LATIN SMALL LETTER T
    u"\u0075": u"\x75",  # LATIN SMALL LETTER U
    u"\u0076": u"\x76",  # LATIN SMALL LETTER V
    u"\u0077": u"\x77",  # LATIN SMALL LETTER W
    u"\u0078": u"\x78",  # LATIN SMALL LETTER X
    u"\u0079": u"\x79",  # LATIN SMALL LETTER Y
    u"\u007A": u"\x7A",  # LATIN SMALL LETTER Z
    u"\u007B": u"\x7B",  # LEFT CURLY BRACKET
    u"\u007C": u"\x7C",  # VERTICAL LINE
    u"\u007D": u"\x7D",  # RIGHT CURLY BRACKET
    u"\u007E": u"\x7E",  # TILDE
    u"\u007F": u"\x7F",  # DELETE
    u"\u20AC": u"\x80",  # EURO SIGN
    u"\u201A": u"\x82",  # SINGLE LOW-9 QUOTATION MARK
    u"\u0192": u"\x83",  # LATIN SMALL LETTER F WITH HOOK
    u"\u201E": u"\x84",  # DOUBLE LOW-9 QUOTATION MARK
    u"\u2026": u"\x85",  # HORIZONTAL ELLIPSIS
    u"\u2020": u"\x86",  # DAGGER
    u"\u2021": u"\x87",  # DOUBLE DAGGER
    u"\u02C6": u"\x88",  # MODIFIER LETTER CIRCUMFLEX ACCENT
    u"\u2030": u"\x89",  # PER MILLE SIGN
    u"\u0160": u"\x8A",  # LATIN CAPITAL LETTER S WITH CARON
    u"\u2039": u"\x8B",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    u"\u0152": u"\x8C",  # LATIN CAPITAL LIGATURE OE
    u"\u017D": u"\x8E",  # LATIN CAPITAL LETTER Z WITH CARON
    u"\u2018": u"\x91",  # LEFT SINGLE QUOTATION MARK
    u"\u2019": u"\x92",  # RIGHT SINGLE QUOTATION MARK
    u"\u201C": u"\x93",  # LEFT DOUBLE QUOTATION MARK
    u"\u201D": u"\x94",  # RIGHT DOUBLE QUOTATION MARK
    u"\u2022": u"\x95",  # BULLET
    u"\u2013": u"\x96",  # EN DASH
    u"\u2014": u"\x97",  # EM DASH
    u"\u02DC": u"\x98",  # SMALL TILDE
    u"\u2122": u"\x99",  # TRADE MARK SIGN
    u"\u0161": u"\x9A",  # LATIN SMALL LETTER S WITH CARON
    u"\u203A": u"\x9B",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    u"\u0153": u"\x9C",  # LATIN SMALL LIGATURE OE
    u"\u017E": u"\x9E",  # LATIN SMALL LETTER Z WITH CARON
    u"\u0178": u"\x9F",  # LATIN CAPITAL LETTER Y WITH DIAERESIS
    u"\u00A0": u"\xA0",  # NO-BREAK SPACE
    u"\u00A1": u"\xA1",  # INVERTED EXCLAMATION MARK
    u"\u00A2": u"\xA2",  # CENT SIGN
    u"\u00A3": u"\xA3",  # POUND SIGN
    u"\u00A4": u"\xA4",  # CURRENCY SIGN
    u"\u00A5": u"\xA5",  # YEN SIGN
    u"\u00A6": u"\xA6",  # BROKEN BAR
    u"\u00A7": u"\xA7",  # SECTION SIGN
    u"\u00A8": u"\xA8",  # DIAERESIS
    u"\u00A9": u"\xA9",  # COPYRIGHT SIGN
    u"\u00AA": u"\xAA",  # FEMININE ORDINAL INDICATOR
    u"\u00AB": u"\xAB",  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    u"\u00AC": u"\xAC",  # NOT SIGN
    u"\u00AD": u"\xAD",  # SOFT HYPHEN
    u"\u00AE": u"\xAE",  # REGISTERED SIGN
    u"\u00AF": u"\xAF",  # MACRON
    u"\u00B0": u"\xB0",  # DEGREE SIGN
    u"\u00B1": u"\xB1",  # PLUS-MINUS SIGN
    u"\u00B2": u"\xB2",  # SUPERSCRIPT TWO
    u"\u00B3": u"\xB3",  # SUPERSCRIPT THREE
    u"\u00B4": u"\xB4",  # ACUTE ACCENT
    u"\u00B5": u"\xB5",  # MICRO SIGN
    u"\u00B6": u"\xB6",  # PILCROW SIGN
    u"\u00B7": u"\xB7",  # MIDDLE DOT
    u"\u00B8": u"\xB8",  # CEDILLA
    u"\u00B9": u"\xB9",  # SUPERSCRIPT ONE
    u"\u00BA": u"\xBA",  # MASCULINE ORDINAL INDICATOR
    u"\u00BB": u"\xBB",  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    u"\u00BC": u"\xBC",  # VULGAR FRACTION ONE QUARTER
    u"\u00BD": u"\xBD",  # VULGAR FRACTION ONE HALF
    u"\u00BE": u"\xBE",  # VULGAR FRACTION THREE QUARTERS
    u"\u00BF": u"\xBF",  # INVERTED QUESTION MARK
    u"\u00C0": u"\xC0",  # LATIN CAPITAL LETTER A WITH GRAVE
    u"\u00C1": u"\xC1",  # LATIN CAPITAL LETTER A WITH ACUTE
    u"\u00C2": u"\xC2",  # LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    u"\u00C3": u"\xC3",  # LATIN CAPITAL LETTER A WITH TILDE
    u"\u00C4": u"\xC4",  # LATIN CAPITAL LETTER A WITH DIAERESIS
    u"\u00C5": u"\xC5",  # LATIN CAPITAL LETTER A WITH RING ABOVE
    u"\u00C6": u"\xC6",  # LATIN CAPITAL LETTER AE
    u"\u00C7": u"\xC7",  # LATIN CAPITAL LETTER C WITH CEDILLA
    u"\u00C8": u"\xC8",  # LATIN CAPITAL LETTER E WITH GRAVE
    u"\u00C9": u"\xC9",  # LATIN CAPITAL LETTER E WITH ACUTE
    u"\u00CA": u"\xCA",  # LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    u"\u00CB": u"\xCB",  # LATIN CAPITAL LETTER E WITH DIAERESIS
    u"\u00CC": u"\xCC",  # LATIN CAPITAL LETTER I WITH GRAVE
    u"\u00CD": u"\xCD",  # LATIN CAPITAL LETTER I WITH ACUTE
    u"\u00CE": u"\xCE",  # LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    u"\u00CF": u"\xCF",  # LATIN CAPITAL LETTER I WITH DIAERESIS
    u"\u00D0": u"\xD0",  # LATIN CAPITAL LETTER ETH
    u"\u00D1": u"\xD1",  # LATIN CAPITAL LETTER N WITH TILDE
    u"\u00D2": u"\xD2",  # LATIN CAPITAL LETTER O WITH GRAVE
    u"\u00D3": u"\xD3",  # LATIN CAPITAL LETTER O WITH ACUTE
    u"\u00D4": u"\xD4",  # LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    u"\u00D5": u"\xD5",  # LATIN CAPITAL LETTER O WITH TILDE
    u"\u00D6": u"\xD6",  # LATIN CAPITAL LETTER O WITH DIAERESIS
    u"\u00D7": u"\xD7",  # MULTIPLICATION SIGN
    u"\u00D8": u"\xD8",  # LATIN CAPITAL LETTER O WITH STROKE
    u"\u00D9": u"\xD9",  # LATIN CAPITAL LETTER U WITH GRAVE
    u"\u00DA": u"\xDA",  # LATIN CAPITAL LETTER U WITH ACUTE
    u"\u00DB": u"\xDB",  # LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    u"\u00DC": u"\xDC",  # LATIN CAPITAL LETTER U WITH DIAERESIS
    u"\u00DD": u"\xDD",  # LATIN CAPITAL LETTER Y WITH ACUTE
    u"\u00DE": u"\xDE",  # LATIN CAPITAL LETTER THORN
    u"\u00DF": u"\xDF",  # LATIN SMALL LETTER SHARP S
    u"\u00E0": u"\xE0",  # LATIN SMALL LETTER A WITH GRAVE
    u"\u00E1": u"\xE1",  # LATIN SMALL LETTER A WITH ACUTE
    u"\u00E2": u"\xE2",  # LATIN SMALL LETTER A WITH CIRCUMFLEX
    u"\u00E3": u"\xE3",  # LATIN SMALL LETTER A WITH TILDE
    u"\u00E4": u"\xE4",  # LATIN SMALL LETTER A WITH DIAERESIS
    u"\u00E5": u"\xE5",  # LATIN SMALL LETTER A WITH RING ABOVE
    u"\u00E6": u"\xE6",  # LATIN SMALL LETTER AE
    u"\u00E7": u"\xE7",  # LATIN SMALL LETTER C WITH CEDILLA
    u"\u00E8": u"\xE8",  # LATIN SMALL LETTER E WITH GRAVE
    u"\u00E9": u"\xE9",  # LATIN SMALL LETTER E WITH ACUTE
    u"\u00EA": u"\xEA",  # LATIN SMALL LETTER E WITH CIRCUMFLEX
    u"\u00EB": u"\xEB",  # LATIN SMALL LETTER E WITH DIAERESIS
    u"\u00EC": u"\xEC",  # LATIN SMALL LETTER I WITH GRAVE
    u"\u00ED": u"\xED",  # LATIN SMALL LETTER I WITH ACUTE
    u"\u00EE": u"\xEE",  # LATIN SMALL LETTER I WITH CIRCUMFLEX
    u"\u00EF": u"\xEF",  # LATIN SMALL LETTER I WITH DIAERESIS
    u"\u00F0": u"\xF0",  # LATIN SMALL LETTER ETH
    u"\u00F1": u"\xF1",  # LATIN SMALL LETTER N WITH TILDE
    u"\u00F2": u"\xF2",  # LATIN SMALL LETTER O WITH GRAVE
    u"\u00F3": u"\xF3",  # LATIN SMALL LETTER O WITH ACUTE
    u"\u00F4": u"\xF4",  # LATIN SMALL LETTER O WITH CIRCUMFLEX
    u"\u00F5": u"\xF5",  # LATIN SMALL LETTER O WITH TILDE
    u"\u00F6": u"\xF6",  # LATIN SMALL LETTER O WITH DIAERESIS
    u"\u00F7": u"\xF7",  # DIVISION SIGN
    u"\u00F8": u"\xF8",  # LATIN SMALL LETTER O WITH STROKE
    u"\u00F9": u"\xF9",  # LATIN SMALL LETTER U WITH GRAVE
    u"\u00FA": u"\xFA",  # LATIN SMALL LETTER U WITH ACUTE
    u"\u00FB": u"\xFB",  # LATIN SMALL LETTER U WITH CIRCUMFLEX
    u"\u00FC": u"\xFC",  # LATIN SMALL LETTER U WITH DIAERESIS
    u"\u00FD": u"\xFD",  # LATIN SMALL LETTER Y WITH ACUTE
    u"\u00FE": u"\xFE",  # LATIN SMALL LETTER THORN
    u"\u00FF": u"\xFF"  # LATIN SMALL LETTER Y WITH DIAERESIS
}

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = []
    for token in tokens:
        if token.isalpha():
            continue
        elif token.lower() not in stopwords:
            filtered_tokens.append(stemmer.stem(token.lower()))
    return filtered_tokens

def tokenize_and_stem2(text):
    # global  map_check_stem
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if not (re.search('[=/]', token)):
                #        if (not(token in map_check_stem.keys())):
                #             map_check_token[token] = 1
                filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')

def spacy_tokenize(text):
    lda_tokens = []
    tokens = parser(doubledecode(text,False))

    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)

    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text(text):
    if not (isinstance(text, float) and math.isnan(text)) and len(text)>0:
        tokens = spacy_tokenize(text)
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in en_stop]
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [get_lemma2(token) for token in tokens]
        #print(type(tokens))
        return tokens


