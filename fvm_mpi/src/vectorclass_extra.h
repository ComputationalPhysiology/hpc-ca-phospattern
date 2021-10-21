#include "vectorclass.h"

#if INSTRSET > 8

#define VECSIZE 8
typedef Vec8d Vecnd;
typedef Vec8db Vecndb;
typedef Vec8i Vecni;

static inline Vec8i get_index(const domain_id *index) {
  return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)index));
}

static Vec8i get_index2(const domain_id *index, __m256i offsetv) {
  __m256i v = _mm256_i32gather_epi32((int const*)index, offsetv,1);
  v = _mm256_and_si256(v, _mm256_set1_epi32(0x7f));
  return v;
}

static inline Vec8d lookupi(Vec8i const &idx, double *p, Vec8db mask) {
  return _mm512_mask_i32gather_pd(_mm512_setzero_pd(), mask, idx, p, 8);
}

static inline Vec8d lookupi(Vec8i const &idx, double *p) {
  return _mm512_i32gather_pd(idx, p, 8);
}

static inline Vec8d load(double *p, Vec8db mask) {
  return _mm512_mask_loadu_pd(_mm512_setzero_pd(), mask, p);
}

static inline void store(double *p, Vec8db mask, Vec8d const &a) {
  _mm512_mask_storeu_pd(p, mask, a);
}

static inline void scatter(double *p, Vec8i const &idx, Vec8d const &a) {
  _mm512_i32scatter_pd(p, idx, a, 8);
}

#elif INSTRSET >= 8

#define VECSIZE 4
typedef Vec4d Vecnd;
typedef Vec4db Vecndb;
typedef Vec4i Vecni;

static inline Vec4i get_index(const domain_id *index) {
  return _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)index));
}

static Vec4i get_index2(const domain_id *index, __m128i offsetv) {
  __m128i v = _mm_i32gather_epi32((int const*)index, offsetv,1);
  v = _mm_and_si128(v, _mm_set1_epi32(0x7f));
  return v;
}

static inline Vec4d lookupi(Vec4i const &idx, double *p, Vec4db mask) {
  return _mm256_mask_i32gather_pd(_mm256_setzero_pd(), p, idx, mask, 8);
}

static inline Vec4d lookupi(Vec4i const &idx, double *p) {
  return _mm256_i32gather_pd(p, idx, 8);
}

static inline Vec4d load(double *p, Vec4db mask) {
  return _mm256_maskload_pd(p, _mm256_castpd_si256(mask));
}

static inline void store(double *p, Vec4db mask, Vec4d const &a) {
  _mm256_maskstore_pd(p, _mm256_castpd_si256(mask), a);
}

static inline void scatter(double *p, Vec4i const &idx, Vec4d const &a) {
  double t[VECSIZE]; 
  a.store(t);
  int32_t idxi[VECSIZE];
  idx.store(idxi);
  for(int i=0; i<VECSIZE; i++) p[idx[i]] = t[i];
}

#elif INSTRSET >= 7

#define VECSIZE 4
typedef Vec4d Vecnd;
typedef Vec4db Vecndb;
typedef Vec4i Vecni;

static inline Vec4i get_index(const domain_id *index) {
  return _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)index));
}

static inline Vec4d lookupi(Vec4i const &idx, double *p, Vec4db mask) {
  double t[VECSIZE];
  int32_t idxi[VECSIZE];
  int64_t tmask[VECSIZE];
  idx.store(idxi);
  Vec4q(mask).store(tmask);
  for(int i=0; i<VECSIZE; i++) t[i] = tmask[i] ? p[idxi[i]] : 0;
  return Vec4d().load(t);
}

static inline Vec4d lookupi(Vec4i const &idx, double *p) {
  double t[VECSIZE];
  int32_t idxi[VECSIZE];
  idx.store(idxi);
  for(int i=0; i<VECSIZE; i++) t[i] = p[idxi[i]];
  return Vec4d().load(t);
}

static inline Vec4d load(double *p, Vec4db mask) {
  return _mm256_maskload_pd(p, _mm256_castpd_si256(mask));
}

static inline void store(double *p, Vec4db mask, Vec4d const &a) {
  _mm256_maskstore_pd(p, _mm256_castpd_si256(mask), a);
}

#else

#define VECSIZE 2
typedef Vec2d Vecnd;
typedef Vec2db Vecndb;
typedef Vec4i Vecni;

static inline Vec4i get_index(const domain_id *index) {
  return _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)index));
}

static inline Vec2d lookupi(Vec4i const &idx, double *p, Vec2db mask) {
  double t[VECSIZE];
  int32_t idxi[2*VECSIZE];
  int64_t tmask[VECSIZE];
  idx.store(idxi);
  Vec2q(mask).store(tmask);
  for(int i=0; i<VECSIZE; i++) t[i] = tmask[i] ? p[idxi[i]] : 0;
  return Vec2d().load(t);
}

static inline Vec2d lookupi(Vec4i const &idx, double *p) {
  double t[VECSIZE];
  int32_t idxi[2*VECSIZE];
  idx.store(idxi);
  for(int i=0; i<VECSIZE; i++) t[i] = p[idxi[i]];
  return Vec2d().load(t);
}

static inline Vec2d load(double *p, Vec2db mask) {
  return select(mask, Vec2d().load(p), 0);
}

static inline void store(double *p, Vec2db mask, Vec2d const &a) {
  double t[VECSIZE]; 
  int64_t tmask[VECSIZE];
  a.store(t);
  Vec2q(mask).store(tmask);
  for(int i=0; i<VECSIZE; i++) if(tmask[i]) p[i] = t[i];
}
#endif

