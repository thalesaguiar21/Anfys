from speech import utils as sut
from data import utils as dut
# import pdb

prompts = dut.get_expected_phns()
runs = dut.get_phns_from_run()

per = 0
total = 0
for run in runs:
    for i in xrange(460):
        res_word = sut.get_phn('fsew0_4', *run[i])
        wlen = len(res_word)
        if wlen != len(prompts[i + 1]):
            print i
            print res_word
            print prompts[i + 1]
            raise ValueError('Different length of words!')
        tmp_pmpt = prompts[i + 1][:]
        ldword = sut.levenshtein_distance(tmp_pmpt, res_word)
        # pdb.set_trace()
        per += ldword / float(wlen)
        total += 1
print per / total
# f.write(str(per) + '\n')
