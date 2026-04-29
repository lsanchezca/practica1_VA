#!/usr/bin/env python3
import os
import csv
from evaluar_resultados import load_results_file, BoundingBox, bboxes_overlap

"""
Genera 'estadisticas_por_imagen_from_evalres.csv' reutilizando funciones de evaluar_resultados.py
Guarda por imagen: imagen,detecciones,gt,tp,fp,fn,precision,recall,f1,score_promedio
"""

def analyze(detections_file='resultado.txt', test_path='test_detection', out_csv='estadisticas_por_imagen_from_evalres.csv', iou_thr=0.5):
    # Cargar detecciones (archivo detections_file) y GT (test_path/gt.txt)
    det_images, det_dbboxes = load_results_file(detections_file, test_path, load_images=False)
    gt_images, gt_dbboxes = load_results_file(os.path.join(test_path, 'gt.txt'), test_path, load_images=False)

    all_images = sorted(set(list(det_dbboxes.keys()) + list(gt_dbboxes.keys())))
    rows = []

    for img in all_images:
        dets = det_dbboxes.get(img, [])
        gts = gt_dbboxes.get(img, [])

        # ordenar detecciones por score descendente
        dets_sorted = sorted(dets, key=lambda x: x.score, reverse=True)
        matched = [False] * len(gts)
        tp = 0
        fp = 0

        for det in dets_sorted:
            best_ov = 0.0
            best_idx = -1
            for gi, gt in enumerate(gts):
                if matched[gi]:
                    continue
                ov = bboxes_overlap(gt, det, ig=(gt.class_id == -1))
                if ov > best_ov:
                    best_ov = ov
                    best_idx = gi

            if best_ov > iou_thr:
                # Si la GT es region de ignore (class_id == -1) no contamos como TP
                if gts[best_idx].class_id != -1 and not matched[best_idx]:
                    tp += 1
                    matched[best_idx] = True
                else:
                    fp += 1
            else:
                fp += 1

        fn = sum(1 for gi, gt in enumerate(gts) if (gt.class_id != -1 and not matched[gi]))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        score_mean = sum(d.score for d in dets_sorted) / len(dets_sorted) if dets_sorted else 0.0

        rows.append({
            'imagen': img,
            'detecciones': len(dets_sorted),
            'gt': len([g for g in gts if g.class_id != -1]),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'score_promedio': round(score_mean, 4)
        })

    # ordenar por f1 descendente
    rows.sort(key=lambda x: x['f1'], reverse=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['imagen','detecciones','gt','tp','fp','fn','precision','recall','f1','score_promedio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # resumen
    if rows:
        mejor = rows[0]
        peor = rows[-1]
        print('Mejor imagen: {}  F1={:.3f}  P={:.3f}  R={:.3f}'.format(mejor['imagen'], mejor['f1'], mejor['precision'], mejor['recall']))
        print('Peor imagen: {}  F1={:.3f}  P={:.3f}  R={:.3f}'.format(peor['imagen'], peor['f1'], peor['precision'], peor['recall']))
    print('CSV generado:', out_csv)

if __name__ == "__main__":
    analyze()
