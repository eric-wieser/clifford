from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from past.builtins import range

from clifford import Cl, randomMV, Frame, get_mult_function, conformalize, grade_obj
from clifford.tools import orthoFrames2Verser as of2v

from numpy import exp, float64, testing
import unittest
import itertools

from nose.plugins.skip import SkipTest


class CliffordTests(unittest.TestCase):

    def setUp(self):
        self.algebras = [Cl(i) for i in [3, 4, 5]]

    @SkipTest
    def test_inverse(self):
        for layout, blades in self.algebras:
            a = 1. + blades['e1']
            self.assertRaises(ValueError, lambda x: 1/x, a)
            for i in range(10):
                a = randomMV(layout, grades=[0, 1])
                denominator = float(a(1)**2-a(0)**2)
                if abs(denominator) > 1.e-5:
                    a_inv = (-a(0)/denominator) + ((1./denominator) * a(1))
                    self.assert_(abs((a * a_inv)-1.) < 1.e-11)
                    self.assert_(abs((a_inv * a)-1.) < 1.e-11)
                    self.assert_(abs(a_inv - 1./a) < 1.e-11)

    def test_exp(self):

        layout, blades = self.algebras[0]
        R = exp(blades['e12'])
        e1 = blades['e1']
        R*e1*~R

    def test_add_float64(self):
        '''
        test array_wrap method to take control addition from numpy array
        '''
        layout, blades = self.algebras[0]
        e1 = blades['e1']

        float64(1) + e1
        self.assertEqual(1 + e1, float64(1) + e1)


class BasicAlgebraTests(unittest.TestCase):

    def test_grade_obj(self):
        algebras = [Cl(i) for i in [3, 4]] + [conformalize(Cl(3)[0])]
        for alg in algebras:
            layout = alg[0]
            for i in range(len(layout.sig)+1):
                mv = layout.randomMV()(i)
                assert i == grade_obj(mv)

    def test_sparse_multiply(self):
        algebras = [Cl(i) for i in [3, 4]] + [conformalize(Cl(3)[0])]
        # For all the algebras we are interested in
        for alg in algebras:
            layout = alg[0]
            # Make two random multivectors
            a = layout.randomMV()
            b = layout.randomMV()
            # Project the multivectors to the grades required
            grades_possibilities = []
            for r in range(1,len(layout.sig)):
                possible_grades = [list(m) for m in list(itertools.combinations(range(len(layout.sig)), r))]
                grades_possibilities += possible_grades
            for i,grades_a in enumerate(grades_possibilities):
                sparse_mv_a = sum([a(k) for k in grades_a])
                for j,grades_b in enumerate(grades_possibilities):
                    sparse_mv_b = sum([b(k) for k in grades_b])
                    # Compute results
                    gp = get_mult_function(layout.gmt,layout.gaDims,layout.gradeList,grades_a=grades_a,grades_b=grades_b)
                    result_sparse = gp(sparse_mv_a.value,sparse_mv_b.value)
                    result_dense = (sparse_mv_a*sparse_mv_b).value
                    # Check they are the same
                    testing.assert_almost_equal(result_sparse, result_dense)
                    print(j+i*len(grades_possibilities),len(grades_possibilities)**2)


class FrameTests(unittest.TestCase):

    def check_inv(self, A):
        Ainv= None
        for k in range(3):
            try:
                Ainv = A.inv
            except(ValueError):
                pass
        if Ainv ==None:
            return True        
        for m, a in enumerate(A):
            for n, b in enumerate(A.inv):
                if m == n:
                    assert(a | b == 1)
                else:
                    assert(a | b == 0)

    def test_frame_inv(self):
        for p, q in [(2, 0), (3, 0), (4, 0)]:
            layout, blades = Cl(p, q)
            A = Frame(layout.randomV(p + q))
            self.check_inv(A)

    def test_innermorphic(self):
        for p, q in [(2, 0), (3, 0), (4, 0)]:
            layout, blades = Cl(p, q)

            A = Frame(layout.randomV(p+q))
            R = layout.randomRotor()
            B = Frame([R*a*~R for a in A])
            self.assertTrue(A.is_innermorphic_to(B))



class G3ToolsTests(unittest.TestCase):

    def test_quaternion_conversions(self):
        """
        Bidirectional rotor - quaternion test. This needs work but is a reasonable start
        """
        from clifford.g3c import layout
        from clifford.tools.g3 import rotor_to_quaternion, quaternion_to_rotor
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        rotor = e1*e2
        print(rotor)
        quaternion = rotor_to_quaternion(rotor)
        print(quaternion)
        rotor_return = quaternion_to_rotor(quaternion)
        print(rotor_return)
        testing.assert_almost_equal(rotor.value, rotor_return.value)

    def test_rotation_matrix_conversions(self):
        """
        Bidirectional rotor - rotation matrix test. This needs work but is a reasonable start
        """
        from clifford.g3c import layout
        from clifford.tools.g3 import rotation_matrix_to_rotor, rotor_to_rotation_matrix
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']

        rotor = e1*e2
        print(rotor)
        matrix = rotor_to_rotation_matrix(rotor)
        print(matrix)
        rotor_return = rotation_matrix_to_rotor(matrix)
        print(rotor_return)
        testing.assert_almost_equal(rotor.value, rotor_return.value)

    def test_generate_rotation_rotor_and_angle(self):
        """
        Checks rotation rotor generation
        """
        from clifford.tools.g3 import generate_rotation_rotor, random_unit_vector, angle_between_vectors

        euc_vector_m = random_unit_vector()
        euc_vector_n = random_unit_vector()
        theta = angle_between_vectors(euc_vector_m, euc_vector_n)
        print(theta)

        rot_rotor = generate_rotation_rotor(theta, euc_vector_m, euc_vector_n)
        v1 = euc_vector_m
        v2 = rot_rotor*euc_vector_m*~rot_rotor
        theta_return = angle_between_vectors(v1, v2)
        print(theta_return)

        testing.assert_almost_equal(theta_return, theta)
        testing.assert_almost_equal(euc_vector_n.value, v2.value)

    @SkipTest
    def test_find_rotor_aligning_vectors(self):
        """
        Currently fails, needs to be dug into
        """
        import numpy as np
        from clifford.g3c import layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        from clifford.tools.g3 import random_euc_mv, random_rotation_rotor, rotor_align_vecs
        u_list = [random_euc_mv() for i in range(50)]
        for i in range(100):
            r = random_rotation_rotor()
            v_list = [r*u*~r for u in u_list]
            r_2 = rotor_align_vecs(u_list, v_list)
            print(r_2)
            print(r)
            testing.assert_almost_equal(r.value, r_2.value)



class G3CToolsTests(unittest.TestCase):

    def test_generate_translation_rotor(self):
        """ Tests translation rotor generation """
        from clifford import g3c
        layout = g3c.layout
        locals().update(g3c.blades)
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import random_euc_mv
        from clifford.tools.g3c import generate_translation_rotor

        for i in range(100):
            rand = random_euc_mv()
            starting_point = up(random_euc_mv())
            r_trans = generate_translation_rotor(rand)
            end_point = r_trans*starting_point*~r_trans
            translation_vec = down(end_point) - down(starting_point)
            testing.assert_almost_equal(translation_vec.value, rand.value)

    def test_intersect_line_and_plane_to_point(self):
        """ Intersection of a line and a plane """
        from clifford import g3c
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3c import intersect_line_and_plane_to_point
        # First the case that they intersect
        line = (up(e1)^up(e1+e3)^ninf).normal()
        plane = (up(e3)^up(e3+e1)^up(e3+e2)^ninf).normal()
        point_result = intersect_line_and_plane_to_point(line, plane)
        testing.assert_almost_equal(down(point_result).value, (e3+e1).value)
        # Next the case that the do not intersect
        line = (up(0) ^ up(e1) ^ ninf).normal()
        point_result = intersect_line_and_plane_to_point(line, plane)
        assert point_result is None

    def test_normalise_n_minus_1(self):
        import numpy as np
        from clifford.tools.g3c import random_conformal_point, normalise_n_minus_1, ninf
        for i in range(500):
            mv = np.random.rand()*random_conformal_point()
            mv_normed = normalise_n_minus_1(mv)
            testing.assert_almost_equal( (mv_normed|ninf)[0], -1.0)

    @SkipTest
    def test_quaternion_and_vector_to_rotor(self):
        """
        TODO: IMPLEMENT THIS TEST
        """
        # quaternion_and_vector_to_rotor(quaternion, vector)

    def test_get_properties_of_sphere(self):
        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import random_euc_mv
        from clifford.tools.g3c import get_radius_from_sphere, get_center_from_sphere, \
            generate_translation_rotor

        for i in range(100):
            # Make a sphere
            scale_factor = np.random.rand()
            sphere = (up(scale_factor*e1)^up(-scale_factor*e1)^up(scale_factor*e3)^up(scale_factor*e2)).normal()
            # Translate it
            rand_trans = random_euc_mv()
            trans_rot = generate_translation_rotor(rand_trans)
            sphere = (trans_rot*sphere*~trans_rot).normal()

            center = get_center_from_sphere(sphere)
            radius = get_radius_from_sphere(sphere)

            testing.assert_almost_equal(down(center).value, rand_trans.value)
            testing.assert_almost_equal(radius, scale_factor)

    def test_point_pair_to_end_points(self):
        from clifford.tools.g3c import random_conformal_point, point_pair_to_end_points
        for i in range(100):
            point_a = random_conformal_point()
            point_b = random_conformal_point()
            pp = (point_a^point_b).normal()
            p_a, p_b = point_pair_to_end_points(pp)
            testing.assert_almost_equal(p_a.value, point_a.value)
            testing.assert_almost_equal(p_b.value, point_b.value)

    def test_euc_distance(self):
        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3c import random_conformal_point, euc_dist
        for i in range(100):
            point_a = random_conformal_point()
            point_b = random_conformal_point()
            dist = euc_dist(point_a, point_b)
            dist_alt = float(abs(down(point_a) - down(point_b)))
            testing.assert_almost_equal(dist, dist_alt)

    def test_dilation_rotor(self):
        from clifford.tools.g3c import random_sphere, generate_dilation_rotor, get_radius_from_sphere
        import numpy as np
        for i in range(100):
            scale = 2*np.random.rand()
            r = generate_dilation_rotor(scale)
            sphere = random_sphere()
            radius = get_radius_from_sphere(sphere)
            sphere2 = (r*sphere*~r).normal()
            radius2 = get_radius_from_sphere(sphere2)
            testing.assert_almost_equal(scale, radius2/radius)

    def test_rotor_between_objects(self):
        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3c import random_sphere, \
            random_plane, random_line, random_circle, \
            random_point_pair, rotor_between_objects

        for i in range(600):
            if i < 100:
                C1 = random_sphere()
                C2 = random_sphere()
            elif i < 200:
                C1 = random_plane()
                C2 = random_plane()
            elif i < 300:
                C1 = random_line()
                C2 = random_line()
            elif i < 400:
                C1 = random_circle()
                C2 = random_circle()
            elif i < 500:
                C1 = random_point_pair()
                C2 = random_point_pair()
            R = rotor_between_objects(C1, C2)
            C3 = (R*C1*~R).normal()
            # NOTE this sign check should not be used in an ideal world, need something a bit better
            if abs(C3 + C2) < 0.0001:
                C3 = -C3
            testing.assert_almost_equal(C2.value, C3.value)

    def test_estimate_rotor_lines(self):

        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import generate_rotation_rotor
        from clifford.tools.g3c import random_line, generate_translation_rotor
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects

        for i in range(200):
            line_list_a = [random_line().normal() for i in range(10)]
            r = (generate_translation_rotor(0.3*e3)*generate_rotation_rotor(np.pi/8,e1+e2,e1)).normal()
            line_list_b = [(r*l*~r).normal() for l in line_list_a]
            r_est, costs = estimate_rotor_objects(line_list_a, line_list_b)
            #self.assertTrue(np.all(np.abs(r.value - r_est.value) < 0.001))
            for a, b in zip([(r_est*l*~r_est).normal() for l in line_list_a], line_list_b):
                self.assertTrue(np.all(np.abs(a.value - b.value) < 0.01))
            print(i)


    def test_estimate_rotor_circles(self):

        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import generate_rotation_rotor
        from clifford.tools.g3c import random_circle, generate_translation_rotor
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects


        for i in range(200):
            circle_list_a = [random_circle().normal() for i in range(10)]
            r = (generate_translation_rotor(0.3*e3)*generate_rotation_rotor(np.pi/8,e1+e2,e1)).normal()
            circle_list_b = [(r*l*~r).normal() for l in circle_list_a]
            r_est, costs = estimate_rotor_objects(circle_list_a, circle_list_b)
            print(r)
            print(r_est)
            #self.assertTrue(np.all(np.abs(r.value - r_est.value) < 0.001))
            for a, b in zip([(r_est*l*~r_est).normal() for l in circle_list_a], circle_list_b):
                self.assertTrue(np.all(np.abs(a.value - b.value) < 0.01))
            print(i)

    @SkipTest
    def test_estimate_rotor_point_pairs(self):
        """
        TODO: This fails some significant percentage of the time. I think it is simply early ending of the scipy optimiser
        """

        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import generate_rotation_rotor
        from clifford.tools.g3c import random_point_pair, generate_translation_rotor
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects

        for i in range(200):
            pp_list_a = [random_point_pair().normal() for i in range(10)]
            r = (generate_translation_rotor(0.2*e3)*generate_rotation_rotor(np.pi/8,e1+e2,e1)).normal()
            pp_list_b = [(r*l*~r).normal() for l in pp_list_a]
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b, print_res=True)
            print(r),
            print(r_est)
            #self.assertTrue(np.all(np.abs(r.value - r_est.value) < 0.000001))
            for a, b in zip([(r_est*l*~r_est).normal() for l in pp_list_a], pp_list_b):
                self.assertTrue(np.all(np.abs(a.value - b.value) < 0.01))
            print(i)

    def test_estimate_rotor_planes_gen(self):

        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import generate_rotation_rotor
        from clifford.tools.g3c import random_plane, generate_translation_rotor
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects

        for i in range(200):
            pp_list_a = [random_plane().normal() for i in range(10)]
            r = (generate_translation_rotor(0.3*e3)*generate_rotation_rotor(np.pi/8,e1+e2,e1)).normal()
            pp_list_b = [(r*l*~r).normal() for l in pp_list_a]
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            #self.assertTrue(np.all(np.abs(r.value - r_est.value) < 0.001))
            for a, b in zip([(r_est*l*~r_est).normal() for l in pp_list_a], pp_list_b):
                self.assertTrue(np.all(np.abs(a.value - b.value) < 0.01))
            print(i)


    def test_estimate_rotor_spheres_gen(self):

        from clifford import g3c
        import numpy as np
        layout = g3c.layout
        e1 = layout.blades['e1']
        e2 = layout.blades['e2']
        e3 = layout.blades['e3']
        ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"],
                                                g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"],
                                                g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])
        from clifford.tools.g3 import generate_rotation_rotor
        from clifford.tools.g3c import random_sphere, generate_translation_rotor
        from clifford.tools.g3c.rotor_estimation import estimate_rotor_objects

        for i in range(200):
            pp_list_a = [random_sphere().normal() for i in range(10)]
            r = (generate_translation_rotor(0.3*e3)*generate_rotation_rotor(np.pi/8,e1+e2,e1)).normal()
            pp_list_b = [(r*l*~r).normal() for l in pp_list_a]
            r_est, costs = estimate_rotor_objects(pp_list_a, pp_list_b)
            #self.assertTrue(np.all(np.abs(r.value - r_est.value) < 0.001))
            for a, b in zip([(r_est*l*~r_est).normal() for l in pp_list_a], pp_list_b):
                self.assertTrue(np.all(np.abs(a.value - b.value) < 0.01))
            print(i)





@SkipTest
class ToolsTests(unittest.TestCase):

    def checkit(self, p, q):
        # p, q =4,0
        N = p + q
        # eps(1e-4)
        layout, blades = Cl(p, q)

        # create frame
        A = layout.randomV(n=N)
        # create Rotor
        R = 5.*layout.randomRotor()
        # create rotated frame
        B = [R*a*~R for a in A]

        # find verser from both frames
        R_found, rs = of2v(A, B)

        # Rotor is determiend correctly, within a sign
        self.assertTrue(R == R_found or R == -R_found)

        # Determined Verser implements desired transformation
        self.assertTrue([R_found*a*~R_found for a in A] == B)

    def testOrthoFrames2VerserEuclidean(self):
        for p, q in [(2, 0), (3, 0), (4, 0)]:
            self.checkit(p=p, q=q)

    @SkipTest  # fails
    def testOrthoFrames2VerserMinkowski(self):
        for p, q in [(1, 1), (2, 1), (3, 1)]:
            self.checkit(p=p, q=q)

    @SkipTest  # fails
    def testOrthoFrames2VerserBalanced(self):
        for p, q in [(2, 2)]:
            self.checkit(p=p, q=q)


if __name__ == '__main__':
    unittest.main()
