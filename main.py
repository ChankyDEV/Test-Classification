import processing.processing as processing

sentences = [
    'First sentence is greate',
    'Another sentece is just better than previous',
    'Previous senteces. are just so much worse than this one',
    'Lorem ipsum dolor sit amet',
    'Maecenas tellus tortor auctor sit amet mi vel',
]

print(processing.process('Test sentence', sentences))
print(processing.process_all(sentences))
