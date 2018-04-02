from BellTwo import BellTwo
from BellThree import BellThree


b2 = BellTwo()
b3 = BellThree()

print 'Tests on Bell Two: '
print 100 * '-'
print 'Membership Degree of 4.0: {}'.format(b2.membership_degree(4.0, 3.0, 2.0))
print 'Derivative at {} relative to \'{}\' {}'.format(4.0, 'a', b2.derivative_at(4.0, 'a', 3.0, 2.0))
print 'Derivative at {} relative to \'{}\' {}'.format(4.0, 'b', b2.derivative_at(4.0, 'b', 3.0, 2.0))
print 'Derivative at {} relative to \'{}\' {}'.format(4.0, 'c', b2.derivative_at(4.0, 'c', 3.0, 2.0))
print ''
print 'Tests on Bell Three'
print 100 * '-'
print 'Membership Degree of 4.0: {}'.format(b3.membership_degree(4, 3, 2, 2))
print 'Derivative at {} relative to \'{}\' {}'.format(4.0, 'a', b3.derivative_at(4, 'a', 3, 2, 2))
print 'Derivative at {} relative to \'{}\' {}'.format(4.0, 'b', b3.derivative_at(4, 'b', 3, 2, 2))
print 'Derivative at {} relative to \'{}\' {}'.format(4.0, 'c', b3.derivative_at(4, 'c', 3, 2, 2))
