#!/usr/bin/env python
import numpy as np
def main():
    L=5
    a = np.arange(L**2).reshape(L,L)
    b = np.pad(a,(1,0),mode='wrap')
    c = np.delete(b,0,0)
    d = np.delete(c,L,1)
    e = np.reshape(d,L**2)
    print a
    print b
    print c
    print d
    print e
main()
