#!/usr/bin/env python
import numpy as np
def main():
    L=3
    a = np.arange(L**3).reshape(L,L,L)
    b = np.pad(a,(1,0),mode='wrap')
    c = np.delete(b,0,0)
    d = np.delete(c,L,1)
    e = np.delete(d,0,2)
    f = np.reshape(e,L**3)
    print a
    print
    print b
    print
    print c
    print
    print d
    print
    print e
    print
    print f
main()
