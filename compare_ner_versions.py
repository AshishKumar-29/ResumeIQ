import json
from collections import Counter
from pathlib import Path

OUTPUTS = Path('outputs')
PAIRS = [
    ('Ashish Kumar', 'Ashish_Kumar_Resume_extracted.json', 'Ashish_Kumar_Resume_v2_extracted.json', [
        'ASHISH KUMAR', 'Software Development Intern', 'ZenZebra', 'Python',
        'Java', 'K.R.M angalamUniversity', 'DPSG .B.NSchool'
    ]),
    ('John Smith', 'Word-Resume-Classic_extracted.json', 'Word-Resume-Classic_v2_extracted.json', [
        'John Smith', 'Customer service cashier', 'SCOTTSDALE Golf course',
        'Phoenix International', 'Adobe Photoshop', 'Adobe Illustrator',
        'Northern Arizona University', 'Brentwood College of Commerce'
    ]),
]

def load_entities(filename):
    path = OUTPUTS / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    return data.get('entities', [])

def matches(entities, phrase):
    phrase_l = phrase.casefold()
    return [e for e in entities if phrase_l in str(e.get('text', '')).casefold()]

for title, old_file, new_file, phrases in PAIRS:
    old_entities = load_entities(old_file)
    new_entities = load_entities(new_file)

    print('\n' + '=' * 72)
    print(title.upper(), '- RAW NER COMPARISON')
    print('=' * 72)
    print(f'v1 entities: {len(old_entities)}')
    print(f'v2 entities: {len(new_entities)}')
    print('\nEntity-label counts:')
    old_counts = Counter(e.get('label', 'UNKNOWN') for e in old_entities)
    new_counts = Counter(e.get('label', 'UNKNOWN') for e in new_entities)
    for label in sorted(set(old_counts) | set(new_counts)):
        print(f'  {label:<14} v1={old_counts[label]:>3}   v2={new_counts[label]:>3}')

    print('\nSelected phrase predictions:')
    for phrase in phrases:
        old_hits = matches(old_entities, phrase)
        new_hits = matches(new_entities, phrase)
        old_labels = ', '.join(sorted({e.get('label', '?') for e in old_hits})) or 'not tagged'
        new_labels = ', '.join(sorted({e.get('label', '?') for e in new_hits})) or 'not tagged'
        status = 'UNCHANGED' if old_labels == new_labels else 'CHANGED'
        print(f'  {phrase}')
        print(f'    v1: {old_labels}')
        print(f'    v2: {new_labels}  [{status}]')

print('\nComparison complete. This compares raw model annotations, not the final post-processed fields.')
