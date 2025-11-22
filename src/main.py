from normalize import normalize

def main():
    text = "שִׁירוֹתִים בַּלִשכַּה וֵסִרטוֹן אֵחַד שֵהוּדלַף"
    normalized = normalize(text)
    print(normalized)

if __name__ == "__main__":
    main()
