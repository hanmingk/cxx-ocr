#pragma once

#include <include/utils.h>
#include <include/flags.h>

#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct PPOCR PPOCR;

    PPOCR *init_ppocr(const Flags &flags);

    void release_ppocr(PPOCR *ppocr);

    const char *ocr(PPOCR *ppocr, const char *img_path,
                    OCRPredictResultArray &array,
                    bool det = true, bool rec = true);

    Flags flags_default();

    void release_flags(Flags &flags);

#ifdef __cplusplus
}
#endif