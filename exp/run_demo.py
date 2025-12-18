from lib.kits.hsmr_demo import *
import cv2
import warnings

# Suppress known warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.meshgrid.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*is_fx_tracing.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*timm.models.layers.*')

# ================== Mesh Rendering Utilities ==================

def render_mesh_only(mesh_verts, mesh_faces, image_shape, K4, cam_t, color_bgr=(255, 0, 0), bg_color=(255, 255, 255), subsample=1):
    """Render mesh wireframe on blank canvas."""
    H, W = image_shape
    img = np.full((H, W, 3), bg_color, dtype=np.uint8)
    
    fx, fy, cx, cy = K4
    verts_cam = mesh_verts + cam_t[np.newaxis, :]
    z = np.maximum(verts_cam[:, 2], 0.01)
    x_img = (verts_cam[:, 0] * fx / z + cx).astype(int)
    y_img = (verts_cam[:, 1] * fy / z + cy).astype(int)
    
    for idx, face in enumerate(mesh_faces[::subsample]):
        pts = np.array([[x_img[face[0]], y_img[face[0]]],
                        [x_img[face[1]], y_img[face[1]]],
                        [x_img[face[2]], y_img[face[2]]]],
                       dtype=np.int32)
        
        valid = np.all((pts >= 0) & (pts < [W, H]), axis=1)
        if valid.sum() >= 2:
            for i in range(3):
                p1 = pts[i]
                p2 = pts[(i + 1) % 3]
                if valid[i] and valid[(i + 1) % 3]:
                    cv2.line(img, tuple(p1), tuple(p2), color_bgr, 1)
    
    return img


def save_mesh_visualizations_for_imgs(raw_imgs, det_meta, m_skin, m_skel, outputs_root, dump_data, img_names=None):
    """Generate and save mesh-only visualizations for images."""
    get_logger(brief=True).info(f'📊 Generating mesh visualizations...')
    
    cur_patch_j = 0
    bbx_cs = det_meta['bbx_cs']
    
    for i in range(len(raw_imgs)):
        H, W = raw_imgs[i].shape[:2]
        raw_cx, raw_cy = W / 2, H / 2
        K4_base = np.array([5000, 5000, raw_cx, raw_cy])
        
        n_patch_cur_img = det_meta['n_patch_per_img'][i]
        
        if img_names is not None:
            img_name = img_names[i]
        else:
            img_name = f'image_{i:04d}'
        
        # Render skin mesh (red) - all patches for this image
        skin_v = m_skin['v'][cur_patch_j:cur_patch_j+n_patch_cur_img].cpu().numpy()
        skin_f = m_skin['f'].cpu().numpy()
        
        # For all patches in this image, render with their camera translations
        skin_img = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
        for patch_idx in range(n_patch_cur_img):
            patch_cam_t_raw = dump_data[cur_patch_j + patch_idx]['patch_cam_t']
            if isinstance(patch_cam_t_raw, np.ndarray):
                patch_cam_t = patch_cam_t_raw[0].copy() if len(patch_cam_t_raw.shape) > 1 else patch_cam_t_raw.copy()
            else:
                patch_cam_t = np.array(patch_cam_t_raw[0]) if hasattr(patch_cam_t_raw, '__len__') and len(patch_cam_t_raw) > 1 else np.array(patch_cam_t_raw)
            
            patch_skin_v = skin_v[patch_idx]
            
            # Rescale camera translation based on bounding box (same as visualize_full_img)
            bbx_scale = bbx_cs[cur_patch_j + patch_idx, 2]
            bbx_cx = bbx_cs[cur_patch_j + patch_idx, 0]
            bbx_cy = bbx_cs[cur_patch_j + patch_idx, 1]
            
            patch_cam_t = np.asarray(patch_cam_t).flatten()
            patch_cam_t[2] = patch_cam_t[2] * 256 / bbx_scale
            patch_cam_t[1] += (bbx_cy - raw_cy) / 5000 * patch_cam_t[2]
            patch_cam_t[0] += (bbx_cx - raw_cx) / 5000 * patch_cam_t[2]
            
            skin_img = render_mesh_overlay(skin_img, patch_skin_v, skin_f, K4_base, patch_cam_t, color_bgr=(0, 0, 255))
        
        skin_path = outputs_root / f'{img_name}_mesh-skin.png'
        cv2.imwrite(str(skin_path), skin_img)
        
        # Render skeleton if available (green)
        if m_skel is not None:
            skel_v = m_skel['v'][cur_patch_j:cur_patch_j+n_patch_cur_img].cpu().numpy()
            skel_f = m_skel['f'].cpu().numpy()
            
            skel_img = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
            for patch_idx in range(n_patch_cur_img):
                patch_cam_t_raw = dump_data[cur_patch_j + patch_idx]['patch_cam_t']
                if isinstance(patch_cam_t_raw, np.ndarray):
                    patch_cam_t = patch_cam_t_raw[0].copy() if len(patch_cam_t_raw.shape) > 1 else patch_cam_t_raw.copy()
                else:
                    patch_cam_t = np.array(patch_cam_t_raw[0]) if hasattr(patch_cam_t_raw, '__len__') and len(patch_cam_t_raw) > 1 else np.array(patch_cam_t_raw)
                
                patch_skel_v = skel_v[patch_idx]
                
                bbx_scale = bbx_cs[cur_patch_j + patch_idx, 2]
                bbx_cx = bbx_cs[cur_patch_j + patch_idx, 0]
                bbx_cy = bbx_cs[cur_patch_j + patch_idx, 1]
                
                patch_cam_t = np.asarray(patch_cam_t).flatten()
                patch_cam_t[2] = patch_cam_t[2] * 256 / bbx_scale
                patch_cam_t[1] += (bbx_cy - raw_cy) / 5000 * patch_cam_t[2]
                patch_cam_t[0] += (bbx_cx - raw_cx) / 5000 * patch_cam_t[2]
                
                skel_img = render_mesh_overlay(skel_img, patch_skel_v, skel_f, K4_base, patch_cam_t, color_bgr=(0, 255, 0), subsample=10)
            
            skel_path = outputs_root / f'{img_name}_mesh-skeleton.png'
            cv2.imwrite(str(skel_path), skel_img)
            
            # Combined
            combined_img = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
            for patch_idx in range(n_patch_cur_img):
                patch_cam_t_raw = dump_data[cur_patch_j + patch_idx]['patch_cam_t']
                if isinstance(patch_cam_t_raw, np.ndarray):
                    patch_cam_t = patch_cam_t_raw[0].copy() if len(patch_cam_t_raw.shape) > 1 else patch_cam_t_raw.copy()
                else:
                    patch_cam_t = np.array(patch_cam_t_raw[0]) if hasattr(patch_cam_t_raw, '__len__') and len(patch_cam_t_raw) > 1 else np.array(patch_cam_t_raw)
                
                patch_skin_v = skin_v[patch_idx]
                patch_skel_v = skel_v[patch_idx]
                
                bbx_scale = bbx_cs[cur_patch_j + patch_idx, 2]
                bbx_cx = bbx_cs[cur_patch_j + patch_idx, 0]
                bbx_cy = bbx_cs[cur_patch_j + patch_idx, 1]
                
                patch_cam_t = np.asarray(patch_cam_t).flatten()
                patch_cam_t[2] = patch_cam_t[2] * 256 / bbx_scale
                patch_cam_t[1] += (bbx_cy - raw_cy) / 5000 * patch_cam_t[2]
                patch_cam_t[0] += (bbx_cx - raw_cx) / 5000 * patch_cam_t[2]
                
                combined_img = render_mesh_overlay(combined_img, patch_skin_v, skin_f, K4_base, patch_cam_t, color_bgr=(0, 0, 255))
                combined_img = render_mesh_overlay(combined_img, patch_skel_v, skel_f, K4_base, patch_cam_t, color_bgr=(0, 255, 0), subsample=10)
            
            combined_path = outputs_root / f'{img_name}_mesh-combined.png'
            cv2.imwrite(str(combined_path), combined_img)
        
        cur_patch_j += n_patch_cur_img
    
    get_logger(brief=True).info(f'✅ Mesh visualization images saved')


def render_mesh_overlay(img, mesh_verts, mesh_faces, K4, cam_t, color_bgr=(255, 0, 0), subsample=1):
    """Render mesh overlay on existing image."""
    H, W = img.shape[:2]
    
    # Ensure K4 is unpacked properly
    K4 = np.asarray(K4).flatten()
    fx, fy, cx, cy = K4[0], K4[1], K4[2], K4[3]
    
    # Ensure cam_t is 1D array with 3 elements
    cam_t = np.asarray(cam_t).flatten()
    if len(cam_t) != 3:
        cam_t = cam_t[:3] if len(cam_t) > 3 else np.pad(cam_t, (0, 3-len(cam_t)), 'constant')
    
    try:
        mesh_verts = np.asarray(mesh_verts)
        mesh_faces = np.asarray(mesh_faces)
        
        verts_cam = mesh_verts + cam_t[np.newaxis, :]
        z = np.maximum(verts_cam[:, 2], 0.01)
        x_img = (verts_cam[:, 0] * fx / z + cx).astype(int)
        y_img = (verts_cam[:, 1] * fy / z + cy).astype(int)
        
        for idx, face in enumerate(mesh_faces[::subsample]):
            pts = np.array([[x_img[face[0]], y_img[face[0]]],
                            [x_img[face[1]], y_img[face[1]]],
                            [x_img[face[2]], y_img[face[2]]]],
                           dtype=np.int32)
            
            valid = np.all((pts >= 0) & (pts < [W, H]), axis=1)
            if valid.sum() >= 2:
                for i in range(3):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % 3]
                    if valid[i] and valid[(i + 1) % 3]:
                        cv2.line(img, tuple(p1), tuple(p2), color_bgr, 1)
    except Exception as e:
        get_logger(brief=True).error(f'Error in render_mesh_overlay: {str(e)}, mesh_verts.shape={mesh_verts.shape if "mesh_verts" in locals() else "unknown"}, mesh_faces.shape={mesh_faces.shape if "mesh_faces" in locals() else "unknown"}')
        raise
    
    return img


def main():
    # ⛩️ 0. Preparation.
    args = parse_args()
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    monitor = TimeMonitor()

    # ⛩️ 1. Preprocess.

    with monitor('Data Preprocessing'):
        with monitor('Load Inputs'):
            raw_imgs, inputs_meta = load_inputs(args)

        with monitor('Detector Initialization'):
            get_logger(brief=True).info('🧱 Building detector.')
            detector = build_detector(
                    batch_size   = args.det_bs,
                    max_img_size = args.det_mis,
                    device       = args.device,
                )

        with monitor('Detecting'):
            get_logger(brief=True).info(f'🖼️ Detecting...')
            detector_outputs = detector(raw_imgs)

        with monitor('Patching & Loading'):
            patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, args.max_instances)  # N * (256, 256, 3)
        if len(patches) == 0:
            get_logger(brief=True).error(f'🚫 No human instance detected. Please ensure the validity of your inputs!')
        get_logger(brief=True).info(f'🔍 Totally {len(patches)} human instances are detected.')


    # ⛩️ 2. Human skeleton and mesh recovery.
    with monitor('Pipeline Initialization'):
        get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)

    with monitor('Recovery'):
        get_logger(brief=True).info(f'🏃 Recovering with B={args.rec_bs}...')
        pd_params, pd_cam_t = [], []
        for bw in asb(total=len(patches), bs_scope=args.rec_bs, enable_tqdm=True):
            patches_i = patches[bw.sid:bw.eid]  # (N, 256, 256, 3)
            patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255  # (N, 256, 256, 3)
            patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)  # (N, 3, 256, 256)
            with torch.no_grad():
                outputs = pipeline(patches_normalized_i)
            pd_params.append({k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()})
            pd_cam_t.append(outputs['pd_cam_t'].detach().cpu().clone())

        pd_params = assemble_dict(pd_params, expand_dim=False)  # [{k:[x]}, {k:[y]}] -> {k:[x, y]}
        pd_cam_t = torch.cat(pd_cam_t, dim=0)
        dump_data = {
                'patch_cam_t' : pd_cam_t.numpy(),
                **{k: v.numpy() for k, v in pd_params.items()},
            }

        get_logger(brief=True).info(f'🤌 Preparing meshes...')
        m_skin, m_skel = prepare_mesh(pipeline, pd_params)
        get_logger(brief=True).info(f'🏁 Done.')


    # ⛩️ 3. Postprocess.
    with monitor('Visualization'):
        if args.ignore_skel:
            m_skel = None
        try:
            results, full_cam_t = visualize_full_img(pd_cam_t, raw_imgs, det_meta, m_skin, m_skel, args.have_caption)
            dump_data['full_cam_t'] = full_cam_t
            visualization_failed = False
        except Exception as e:
            get_logger(brief=True).warning(f'⚠️ Visualization rendering failed: {e}. Saving raw images and mesh data.')
            results = raw_imgs  # Fallback to raw images
            full_cam_t = None
            visualization_failed = True
        
        # Save results (images or video), mesh geometry, and pose data.
        if inputs_meta['type'] == 'video':
            seq_name = f'{pipeline.name}-' + inputs_meta['seq_name']
            # Save video
            save_video(results, outputs_root / f'{seq_name}.mp4')
            get_logger(brief=True).info(f'🎥 Video saved: {seq_name}.mp4')
            
            # Save individual frames as images
            for i, frame in enumerate(results):
                frame_name = f'{seq_name}_frame_{i:04d}.jpg'
                save_img(frame, outputs_root / frame_name)
            get_logger(brief=True).info(f'🖼️ Saved {len(results)} frame images.')
            
            # Dump mesh geometry and pose data for each frame
            dump_results = []
            cur_patch_j = 0
            for i in range(len(raw_imgs)):
                n_patch_cur_img = det_meta['n_patch_per_img'][i]
                dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                # Add mesh geometry
                dump_results_i['mesh_skin_v'] = m_skin['v'][cur_patch_j:cur_patch_j+n_patch_cur_img].cpu().numpy()
                dump_results_i['mesh_skin_f'] = m_skin['f'].cpu().numpy()
                if m_skel is not None:
                    dump_results_i['mesh_skel_v'] = m_skel['v'][cur_patch_j:cur_patch_j+n_patch_cur_img].cpu().numpy()
                    dump_results_i['mesh_skel_f'] = m_skel['f'].cpu().numpy()
                cur_patch_j += n_patch_cur_img
                dump_results.append(dump_results_i)
            np.save(outputs_root / f'{seq_name}_data.npy', dump_results)
            get_logger(brief=True).info(f'💾 Mesh geometry and pose data saved: {seq_name}_data.npy')
        
        if inputs_meta['type'] == 'imgs':
            img_names = [f'{pipeline.name}-{fn.stem}' for fn in inputs_meta['img_fns']]
            # Save images and mesh geometry + pose data
            cur_patch_j = 0
            for i, img_name in enumerate(tqdm(img_names, desc='Saving images and meshes')):
                n_patch_cur_img = det_meta['n_patch_per_img'][i]
                dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                # Add mesh geometry (skin and skeleton vertices/faces)
                dump_results_i['mesh_skin_v'] = m_skin['v'][cur_patch_j:cur_patch_j+n_patch_cur_img].cpu().numpy()
                dump_results_i['mesh_skin_f'] = m_skin['f'].cpu().numpy()
                if m_skel is not None:
                    dump_results_i['mesh_skel_v'] = m_skel['v'][cur_patch_j:cur_patch_j+n_patch_cur_img].cpu().numpy()
                    dump_results_i['mesh_skel_f'] = m_skel['f'].cpu().numpy()
                cur_patch_j += n_patch_cur_img
                # Save rendered/original image
                save_img(results[i], outputs_root / f'{img_name}.jpg')
                # Save mesh geometry and pose data as npz
                np.savez(outputs_root / f'{img_name}_mesh.npz', **dump_results_i)
            get_logger(brief=True).info(f'🖼️ Saved {len(img_names)} images with mesh geometry and pose data.')

        get_logger(brief=True).info(f'🎨 Rendering results are under {outputs_root}.')
        
        # Generate mesh-only visualizations if requested
        if args.save_mesh_vis:
            try:
                # Build dump_data list with all patches
                dump_data_list = []
                cur_patch_j = 0
                for i in range(len(raw_imgs)):
                    n_patch_cur_img = det_meta['n_patch_per_img'][i]
                    for patch_idx in range(n_patch_cur_img):
                        dump_results_i = {k: v[cur_patch_j:cur_patch_j+1] for k, v in dump_data.items()}
                        dump_data_list.append(dump_results_i)
                        cur_patch_j += 1
                
                if inputs_meta['type'] == 'imgs':
                    img_names = [f'{pipeline.name}-{fn.stem}' for fn in inputs_meta['img_fns']]
                    save_mesh_visualizations_for_imgs(raw_imgs, det_meta, m_skin, m_skel, outputs_root, dump_data_list, img_names=img_names)
                else:  # video
                    get_logger(brief=True).info(f'🎥 Generating mesh visualization video...')
                    mesh_frames = []
                    bbx_cs = det_meta['bbx_cs']
                    cur_patch_j = 0
                    
                    # Create frames subdirectory for storing individual rendered frames
                    frames_dir = outputs_root / 'mesh_frames'
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i in tqdm(range(len(raw_imgs)), desc='Rendering mesh overlays'):
                        H, W = raw_imgs[i].shape[:2]
                        raw_cx, raw_cy = W / 2, H / 2
                        K4_base = np.array([5000, 5000, raw_cx, raw_cy])
                        
                        n_patch_cur_img = det_meta['n_patch_per_img'][i]
                        # Start with the original frame
                        mesh_img = raw_imgs[i].copy()
                        
                        for patch_idx in range(n_patch_cur_img):
                            try:
                                patch_cam_t_raw = dump_data_list[cur_patch_j]['patch_cam_t']
                                if isinstance(patch_cam_t_raw, np.ndarray):
                                    patch_cam_t = patch_cam_t_raw[0].copy() if len(patch_cam_t_raw.shape) > 1 else patch_cam_t_raw.copy()
                                else:
                                    patch_cam_t = np.array(patch_cam_t_raw[0]) if hasattr(patch_cam_t_raw, '__len__') and len(patch_cam_t_raw) > 1 else np.array(patch_cam_t_raw)
                                
                                skin_v = m_skin['v'][cur_patch_j].cpu().numpy()
                                skin_f = m_skin['f'].cpu().numpy()
                                
                                # Rescale camera translation based on bounding box (same as visualize_full_img)
                                bbx_scale = bbx_cs[cur_patch_j, 2]  # bounding box scale
                                bbx_cx = bbx_cs[cur_patch_j, 0]   # bounding box center x
                                bbx_cy = bbx_cs[cur_patch_j, 1]   # bounding box center y
                                
                                # Rescale Z and adjust X, Y based on bbox position
                                patch_cam_t = np.asarray(patch_cam_t).flatten()
                                patch_cam_t[2] = patch_cam_t[2] * 256 / bbx_scale  # Scale based on bbox
                                patch_cam_t[1] += (bbx_cy - raw_cy) / 5000 * patch_cam_t[2]  # Adjust Y
                                patch_cam_t[0] += (bbx_cx - raw_cx) / 5000 * patch_cam_t[2]  # Adjust X
                                
                                # Render skin mesh on the actual frame (red)
                                mesh_img = render_mesh_overlay(mesh_img, skin_v, skin_f, K4_base, patch_cam_t, color_bgr=(0, 0, 255))
                                
                                # Render skeleton if available (green)
                                if m_skel is not None:
                                    skel_v = m_skel['v'][cur_patch_j].cpu().numpy()
                                    skel_f = m_skel['f'].cpu().numpy()
                                    mesh_img = render_mesh_overlay(mesh_img, skel_v, skel_f, K4_base, patch_cam_t, color_bgr=(0, 255, 0), subsample=10)
                            except Exception as e:
                                get_logger(brief=True).warning(f'⚠️ Failed to render mesh for patch {cur_patch_j} (frame {i}): {str(e)}')
                            
                            cur_patch_j += 1
                        
                        mesh_frames.append(mesh_img)
                        # Save individual frame
                        frame_path = frames_dir / f'frame_{i:06d}_mesh.png'
                        cv2.imwrite(str(frame_path), mesh_img)
                    
                    mesh_video_path = outputs_root / f'{pipeline.name}-mesh.mp4'
                    save_video(mesh_frames, mesh_video_path)
                    get_logger(brief=True).info(f'✅ Mesh visualization video saved: {pipeline.name}-mesh.mp4')
                    get_logger(brief=True).info(f'📁 Individual mesh frames saved to: mesh_frames/ ({len(mesh_frames)} frames)')
            except Exception as e:
                get_logger(brief=True).warning(f'⚠️ Mesh visualization generation failed: {e}')
                traceback.print_exc()

    get_logger(brief=True).info(f'🎊 Everything is done!')
    monitor.report()


if __name__ == '__main__':
    main()