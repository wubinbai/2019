import thinkdsp
COEFF = 2**(1/12)

C_Hz = 261.63
C_low_Hz = C_Hz/2
E_b_Hz = C_Hz * COEFF**(3)
E_Hz = C_Hz * COEFF**(4)

G_Hz = C_Hz * COEFF**(7)

S1_L = thinkdsp.SinSignal(freq=C_low_Hz)
S1 = C_Sig = thinkdsp.SinSignal(freq=C_Hz,amp=1.0,offset=0)
S4 = E_b_Sig = thinkdsp.SinSignal(freq=E_b_Hz)
S5 = E_Sig = thinkdsp.SinSignal(freq=E_Hz,amp=1.0,offset=0)
S8 = G_Sig = thinkdsp.SinSignal(freq=G_Hz)

SMaj = major_third_Sig = S1_L + S1 + S5 + S8
SMin = minor_third_Sig = S1 + S4 + S8
Wave_maj = major_third_Wave = SMaj.make_wave(duration=0.5)
Wave_min = minor_third_Wave = SMin.make_wave(duration=0.5)

spec_maj = Wave_maj.make_spectrum()
spec_maj.plot()
plt.show()

plt.figure()
Wave_maj.plot()
plt.show()

plt.figure()
Wave_min.plot()
plt.show()


