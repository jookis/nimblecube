"""Prepare the UCI UNIX user command data as a sequence-classification benchmark (BENCHMARKS.md §13).

Source: T. Lane, "UNIX User Data", UCI Machine Learning Repository (id 141), CC BY 4.0:
    https://archive.ics.uci.edu/static/public/141/unix+user+data.zip
Nine token streams from eight users, sanitised, with **SOF** / **EOF** marking shell sessions. Task: given
a session, say which user it came from. Sessions are variable length symbolic sequences, the case the
encoding is supposed to suit (no fixed alignment, no cheap raw distance).

Split: chronological per user (sessions are in date order), first 70% train, last 30% test. That is harder
and more honest than a random split, because user habits drift over the two years.

Writes train.txt / test.txt as "sequences vocab" then one line per session: "label tok tok ...".
Tokens are vocabulary indices built from the TRAINING sessions only; unseen tokens map to a single
"unknown" id, as any real system would have to do.

Run:  python3 seq_prep.py <UNIX_user_data dir> <out dir> [--min-len=5] [--max-len=200]
"""
import sys, os, glob

src, out = sys.argv[1], sys.argv[2]
MINL = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--min-len=')), 5))
MAXL = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--max-len=')), 200))
os.makedirs(out, exist_ok=True)

users, train, test = [], [], []
for f in sorted(glob.glob(os.path.join(src, 'USER*'))):
    if f.endswith('.gz'):
        continue
    label = len(users); users.append(os.path.basename(f))
    sess, cur = [], None
    for t in open(f, errors='replace').read().split():
        if t == '**SOF**':
            cur = []
        elif t == '**EOF**':
            if cur and MINL <= len(cur):
                sess.append(cur[:MAXL])
            cur = None
        elif cur is not None:
            cur.append(t)
    cut = int(len(sess) * 0.7)
    train += [(label, s) for s in sess[:cut]]
    test += [(label, s) for s in sess[cut:]]

vocab = {'<unk>': 0}
for _, s in train:
    for t in s:
        vocab.setdefault(t, len(vocab))

def write(path, rows):
    with open(path, 'w') as fh:
        fh.write(f"{len(rows)} {len(vocab)}\n")
        for label, s in rows:
            fh.write(f"{label} " + " ".join(str(vocab.get(t, 0)) for t in s) + "\n")

write(os.path.join(out, 'train.txt'), train)
write(os.path.join(out, 'test.txt'), test)
unk = sum(1 for _, s in test for t in s if t not in vocab) / max(sum(len(s) for _, s in test), 1)
print(f"{len(users)} users, {len(train)} train and {len(test)} test sessions, vocab {len(vocab)}, "
      f"{unk:.1%} of test tokens unseen in training")
print("per user (train/test): " + ", ".join(
    f"{u} {sum(1 for l, _ in train if l == i)}/{sum(1 for l, _ in test if l == i)}" for i, u in enumerate(users)))
print("median session length: train %d, test %d" % (
    sorted(len(s) for _, s in train)[len(train) // 2], sorted(len(s) for _, s in test)[len(test) // 2]))
