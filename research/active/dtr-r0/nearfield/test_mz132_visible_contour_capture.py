"""UE Rotator input-type regression; no renderer or outcome data required."""
import copy

from mz132_visible_contour_capture import verify_geometry


def test_rotator_float32_input_and_real_pose_error():
    # Observed constructor conversion from the frozen camera specification.
    scene=dict(camera=dict(x=0.,y=0.,z=1.7,pitch=-3.,yaw=.08502610912110925,roll=0.),objects=[])
    spec=dict(rig=dict(width=640,height=360,hfov_deg=70.),background=None,floor=None)
    frame=dict(instance_count=0,renderer_group_sizes=[],evaluator_geometry=dict(
        camera_location_m=[0.,0.,1.7],camera_rotation_deg=[-3.,.08502610772848129,0.],
        capture_hfov_deg=70.,render_target_size=[640,360],native_base_instances=[]))
    assert verify_geometry(frame,scene,spec,dict(native_bounds=[]))['passed']
    perturbed=copy.deepcopy(frame)
    perturbed['evaluator_geometry']['camera_rotation_deg'][1]+=.00001
    result=verify_geometry(perturbed,scene,spec,dict(native_bounds=[]))
    assert not result['passed'] and not result['checks']['camera_rotation']


if __name__=='__main__':
    test_rotator_float32_input_and_real_pose_error()
    print('PASS: constructor float32 accepted; actual pose offset rejected')
