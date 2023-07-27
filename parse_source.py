import xmltodict
import json
import argparse
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg("--code", type=str, required=True),
arg("--name", type=str, required=True),
args = parser.parse_args()

# Parse Opus XML format
with open(f'./{args.code}/bible-uedin/xml/{args.code}/{args.name}.xml') as xml_file:
    data_dict = xmltodict.parse(xml_file.read())
    x = json.dumps(data_dict)
    json_data = json.loads(x)
    sentences = []
    for a in json_data['cesDoc']['text']['body']['div']:
        for b in a['div']:
            try:
                for c in b['seg']:
                    try:
                        sentence_words = []
                        for d in c['s']['w']:
                            word = d['#text']
                            if word in ["!", ".", "?"]:
                                continue
                            sentence_words.append(word)
                        sentence = " ".join(sentence_words)
                        sentences.append(sentence)
                    except:
                        for d in c['s']:
                            sentence_words = []
                            for e in d['w']:
                                try:
                                    word = e['#text']
                                    if word in ["!", ".", "?"]: #For convenience, just remove these here.
                                        continue
                                    sentence_words.append(word)
                                except:
                                    continue
                            sentence = " ".join(sentence_words)
                            sentences.append(sentence)
            except:
                continue
    with open(f'parsed_corpora/{args.name}.txt', 'a', encoding="utf8") as f:
        for s in sentences:
            f.write(f'{s}\n')