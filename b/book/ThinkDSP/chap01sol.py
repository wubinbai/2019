import thinkdsp
f = thinkdsp.read_wave('chords_sample.wav')
seg = f.segment(duration=0.5)
ss = seg.make_spectrum()
ss.plot(high=1500)
ss.low_pass(cutoff=600)
nw = ss.make_wave()
nw.play()

