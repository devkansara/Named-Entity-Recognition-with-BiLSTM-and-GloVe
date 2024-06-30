import sys
import os
import argparse


parser = argparse.ArgumentParser(prog="eval", description="Evaluation of HW4")
parser.add_argument("-g", "--gold", help="gold standard file")
parser.add_argument("-p", "--pred", help="prediction file")
args = parser.parse_args()

pred = args.pred
gold = args.gold
tmp = 'tmp.out'


with open(pred, 'r') as prf, open(gold, 'r') as grf, open(tmp, 'w') as writer:
    for gline in grf:
        pline = prf.readline()

        pline = pline.strip()
        gline = gline.strip()
        
        if len(pline) == 0:
            writer.write('\n')
            continue

        tokens = gline.split()
        # print tokens
        gidx = tokens[0]
        gword = tokens[1]
        glabel = tokens[2]

        tokens = pline.split()
        pidx = tokens[0]
        pword = tokens[1]


        if gidx != pidx:
            print('warning: index mismatch: {}, {}'.format(gidx, pidx))

        if gword != pword:
            print('warning: word mismatch: {}, {}'.format(gword, pword))


        plabel = tokens[2] if len(tokens) == 3 else tokens[3]

        writer.write(gidx + " " + gword + " " + glabel + " " + plabel)
        writer.write('\n')

script = 'conll03eval'
os.system("perl %s < %s" % (script, tmp))
