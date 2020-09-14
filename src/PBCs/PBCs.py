#!/usr/bin/env python
def main():
    L=10
    for i in xrange(L):
    	print (i+1) % L
    for i in xrange(L):
    	print (i+L-1) % L
#    print 10 % 10

main()
