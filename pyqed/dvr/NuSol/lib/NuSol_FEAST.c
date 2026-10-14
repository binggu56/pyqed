#define _POSIX_C_SOURCE 200809L

/*
 * NuSol FEAST sparse solver, based on Eric Polizzi's FEAST example driver.
 *
 * This command-line program processes scientific data supplied by the invoking
 * user. Inputs are bounded and validated before they influence allocation or
 * indexing, and result files are created with owner-only permissions.
 */

#include <errno.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "feast.h"
#include "feast_sparse.h"

#ifdef O_NOFOLLOW
#define NOFOLLOW_FLAG O_NOFOLLOW
#else
#define NOFOLLOW_FLAG 0
#endif

#define MAX_MATRIX_DIM 50000000L
#define MAX_NONZERO 100000000L
#define MAX_SUBSPACE 100000L
#define MAX_VECTOR_ELEMENTS 100000000UL

static int parse_double_arg(const char *text, double *value) {
  char *end = NULL;

  errno = 0;
  *value = strtod(text, &end);
  return errno == 0 && end != text && *end == '\0' && isfinite(*value);
}

static int parse_positive_int_arg(const char *text, long maximum, int *value) {
  char *end = NULL;
  long parsed;

  errno = 0;
  parsed = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed <= 0 ||
      parsed > maximum || parsed > INT_MAX) {
    return 0;
  }
  *value = (int)parsed;
  return 1;
}

static int read_token(FILE *stream, char token[64]) {
  if (fscanf(stream, "%63s", token) != 1) return 0;
  int next = fgetc(stream);
  return (next == EOF || isspace((unsigned char)next)) && !ferror(stream);
}

static int read_long(FILE *stream, long *value) {
  char token[64], *end;
  if (!read_token(stream, token)) return 0;
  errno = 0;
  *value = strtol(token, &end, 10);
  return errno == 0 && end != token && *end == '\0';
}

static int read_double(FILE *stream, double *value) {
  char token[64];
  return read_token(stream, token) && parse_double_arg(token, value);
}

static void *bounded_calloc(size_t count, size_t size) {
  if (count == 0 || count > MAX_VECTOR_ELEMENTS || size == 0 ||
      count > SIZE_MAX / size) {
    return NULL;
  }
  return calloc(count, size);
}

static FILE *open_input(const char *path) {
  int descriptor;
  struct stat metadata;

  if (lstat(path, &metadata) != 0) {
    return NULL;
  }
  if (S_ISLNK(metadata.st_mode)) {
    errno = ELOOP;
    return NULL;
  }
  /* The path is selected by the local CLI caller, not by a remote principal. */
  descriptor = open(path, O_RDONLY | O_NONBLOCK | NOFOLLOW_FLAG); /* lgtm[cpp/path-injection] */
  if (descriptor < 0) {
    return NULL;
  }
  if (fstat(descriptor, &metadata) != 0 || !S_ISREG(metadata.st_mode)) {
    close(descriptor);
    errno = EINVAL;
    return NULL;
  }
  FILE *stream = fdopen(descriptor, "r");
  if (stream == NULL) close(descriptor);
  return stream;
}

static FILE *open_output(const char *path, const struct stat *input, FILE *other) {
  int descriptor;
  struct stat metadata;

  if (lstat(path, &metadata) == 0) {
    if (S_ISLNK(metadata.st_mode)) {
      errno = ELOOP;
      return NULL;
    }
  } else if (errno != ENOENT) {
    return NULL;
  }
  /* NOFOLLOW_FLAG prevents races where supported; 0600 avoids data leakage. */
  descriptor = open(path, O_WRONLY | O_CREAT | O_NONBLOCK | NOFOLLOW_FLAG,
                    S_IRUSR | S_IWUSR); /* lgtm[cpp/path-injection] */
  if (descriptor < 0) {
    return NULL;
  }
  struct stat other_metadata;
  if (fstat(descriptor, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_nlink != 1 ||
      (metadata.st_dev == input->st_dev && metadata.st_ino == input->st_ino) ||
      (other != NULL && (fstat(fileno(other), &other_metadata) != 0 ||
       (metadata.st_dev == other_metadata.st_dev && metadata.st_ino == other_metadata.st_ino))) ||
      fchmod(descriptor, S_IRUSR | S_IWUSR) != 0) {
    close(descriptor);
    errno = EINVAL;
    return NULL;
  }
  FILE *stream = fdopen(descriptor, "w");
  if (stream == NULL) close(descriptor);
  return stream;
}

static void report_errno(const char *operation, const char *path) {
  fprintf(stderr, "NuSol_FEAST: cannot %s '%s': %s\n", operation, path,
          strerror(errno));
}

int main(int argc, char *argv[]) {
  int feastparam[64];
  double epsout;
  int loop;
  char uplo = 'F';
  FILE *matrix_file = NULL;
  FILE *value_file = NULL;
  FILE *vector_file = NULL;
  double *sa = NULL;
  double *sb = NULL;
  int *isa = NULL;
  int *jsa = NULL;
  double *vectors = NULL;
  double *eigenvalues = NULL;
  double *residuals = NULL;
  struct timeval start_time;
  struct timeval end_time;
  struct stat input_metadata;
  int exit_code = EXIT_FAILURE;
  int n;
  int i;
  int k;
  int n_matrix;
  int nnz;
  int m0;
  int m;
  int info;
  int nx;
  int ny;
  int nz;
  long n_input;
  long nnz_input;
  long nx_input;
  long ny_input;
  long nz_input;
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  double zmin;
  double zmax;
  double dx;
  double dy;
  double dz;
  double emin;
  double emax;
  double trace;
  size_t vector_elements;
  long previous_row = 0, previous_column = 0;

  if (argc != 7) {
    fprintf(stderr,
            "usage: %s Emin Emax M0 matrix-file eigenvalues-file "
            "eigenvectors-file\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  if (!parse_double_arg(argv[1], &emin) ||
      !parse_double_arg(argv[2], &emax) || emin >= emax ||
      !parse_positive_int_arg(argv[3], MAX_SUBSPACE, &m0)) {
    fprintf(stderr, "NuSol_FEAST: invalid search interval or subspace size\n");
    goto cleanup;
  }

  matrix_file = open_input(argv[4]);
  if (matrix_file == NULL) {
    report_errno("open matrix input", argv[4]);
    goto cleanup;
  }

  if (fstat(fileno(matrix_file), &input_metadata) != 0 ||
      !read_long(matrix_file, &n_input) || !read_long(matrix_file, &nnz_input) ||
      !read_long(matrix_file, &nx_input) || !read_long(matrix_file, &ny_input) ||
      !read_long(matrix_file, &nz_input) || !read_double(matrix_file, &xmin) ||
      !read_double(matrix_file, &xmax) || !read_double(matrix_file, &ymin) ||
      !read_double(matrix_file, &ymax) || !read_double(matrix_file, &zmin) ||
      !read_double(matrix_file, &zmax) || !read_double(matrix_file, &dx) ||
      !read_double(matrix_file, &dy) || !read_double(matrix_file, &dz)) {
    fprintf(stderr, "NuSol_FEAST: malformed matrix header\n");
    goto cleanup;
  }
  if (n_input <= 0 || n_input > MAX_MATRIX_DIM || nnz_input <= 0 ||
      nnz_input > MAX_NONZERO || nx_input < 0 || nx_input > INT_MAX ||
      ny_input < 0 || ny_input > INT_MAX || nz_input < 0 ||
      nz_input > INT_MAX || m0 > n_input) {
    fprintf(stderr, "NuSol_FEAST: matrix dimensions are outside safe limits\n");
    goto cleanup;
  }

  n_matrix = (int)n_input;
  nnz = (int)nnz_input;
  nx = (int)nx_input;
  ny = (int)ny_input;
  nz = (int)nz_input;
  if ((size_t)n_matrix > MAX_VECTOR_ELEMENTS / (size_t)m0) {
    fprintf(stderr, "NuSol_FEAST: eigenvector workspace is too large\n");
    goto cleanup;
  }
  vector_elements = (size_t)n_matrix * (size_t)m0;

  sa = bounded_calloc((size_t)nnz, sizeof(*sa));
  sb = bounded_calloc((size_t)nnz, sizeof(*sb));
  isa = bounded_calloc((size_t)n_matrix + 1U, sizeof(*isa));
  jsa = bounded_calloc((size_t)nnz, sizeof(*jsa));
  eigenvalues = bounded_calloc((size_t)m0, sizeof(*eigenvalues));
  residuals = bounded_calloc((size_t)m0, sizeof(*residuals));
  vectors = bounded_calloc(vector_elements, sizeof(*vectors));
  if (sa == NULL || sb == NULL || isa == NULL || jsa == NULL ||
      eigenvalues == NULL || residuals == NULL || vectors == NULL) {
    fprintf(stderr, "NuSol_FEAST: unable to allocate solver workspace\n");
    goto cleanup;
  }

  isa[0] = 1;
  for (k = 0; k < nnz; ++k) {
    long row;
    long column;

    if (!read_long(matrix_file, &row) || !read_long(matrix_file, &column) ||
        !read_double(matrix_file, &sa[k]) || !read_double(matrix_file, &sb[k])) {
      fprintf(stderr, "NuSol_FEAST: malformed matrix entry %d\n", k + 1);
      goto cleanup;
    }
    if (row < 1 || row > n_matrix || column < 1 || column > n_matrix) {
      fprintf(stderr, "NuSol_FEAST: matrix index out of range at entry %d\n",
              k + 1);
      goto cleanup;
    }
    if (row < previous_row || (row == previous_row && column <= previous_column)) {
      fprintf(stderr, "NuSol_FEAST: matrix entries must be sorted and unique\n");
      goto cleanup;
    }
    previous_row = row;
    previous_column = column;
    jsa[k] = (int)column;
    ++isa[(int)row];
  }
  for (i = 1; i <= n_matrix; ++i) {
    isa[i] += isa[i - 1];
  }
  int trailing;
  do { trailing = fgetc(matrix_file); } while (trailing != EOF && isspace((unsigned char)trailing));
  if (trailing != EOF || ferror(matrix_file)) {
    fprintf(stderr, "NuSol_FEAST: unexpected data after matrix entries\n");
    goto cleanup;
  }

  if (fclose(matrix_file) != 0) {
    matrix_file = NULL;
    fprintf(stderr, "NuSol_FEAST: failed to close matrix input\n");
    goto cleanup;
  }
  matrix_file = NULL;

  value_file = open_output(argv[5], &input_metadata, NULL);
  if (value_file == NULL) {
    report_errno("create eigenvalue output", argv[5]);
    goto cleanup;
  }
  vector_file = open_output(argv[6], &input_metadata, value_file);
  if (vector_file == NULL) {
    report_errno("create eigenvector output", argv[6]);
    goto cleanup;
  }
  if (ftruncate(fileno(value_file), 0) != 0 || ftruncate(fileno(vector_file), 0) != 0) {
    fprintf(stderr, "NuSol_FEAST: cannot truncate result files\n");
    goto cleanup;
  }

  printf("sparse matrix size %d, nnz %d\n", n_matrix, nnz);
  gettimeofday(&start_time, NULL);
  feastinit(feastparam);
  feastparam[0] = 1;
  feastparam[1] = 32;
  feastparam[2] = 14;
  feastparam[3] = 30;
  dfeast_scsrgv(&uplo, &n_matrix, sa, isa, jsa, sb, isa, jsa, feastparam,
                &epsout, &loop, &emin, &emax, &m0, eigenvalues, vectors, &m,
                residuals, &info);
  gettimeofday(&end_time, NULL);

  printf("FEAST OUTPUT INFO %d\n", info);
  if (info != 0 || m < 0 || m > m0) {
    if (info == 0) {
      fprintf(stderr, "NuSol_FEAST: FEAST returned an invalid mode count\n");
    }
    goto cleanup;
  }

  printf("SIMULATION TIME %f\n",
         (end_time.tv_sec - start_time.tv_sec) * 1.0 +
             (end_time.tv_usec - start_time.tv_usec) * 0.000001);
  printf("# Search interval [Emin,Emax] %.15e %.15e\n", emin, emax);
  printf("# mode found/subspace %d %d\n", m, m0);
  printf("# iterations %d\n", loop);
  trace = 0.0;
  for (i = 0; i < m; ++i) {
    trace += eigenvalues[i];
  }
  printf("TRACE %.15e\n", trace);
  printf("Relative error on the Trace %.15e\n", epsout);
  printf("Eigenvalues/Residuals\n");
  printf("%d %d %d %f %f %f %f %f %f\n", nx, ny, nz, xmin, xmax, ymin,
         ymax, zmin, zmax);
  fprintf(vector_file, "%d %d %d %f %f %f %f %f %f %f %f %f\n", nx, ny,
          nz, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz);
  for (i = 0; i < m; ++i) {
    printf("   %d %.15e %.15e\n", i, eigenvalues[i], residuals[i]);
    fprintf(value_file, "%.15e %.15e\n", eigenvalues[i], residuals[i]);
    for (n = 0; n < n_matrix; ++n) {
      fprintf(vector_file, "%.15e ", vectors[(size_t)n_matrix * (size_t)i +
                                             (size_t)n]);
    }
    fputc('\n', vector_file);
  }

  if (fflush(value_file) != 0 || fflush(vector_file) != 0) {
    fprintf(stderr, "NuSol_FEAST: failed to write result files\n");
    goto cleanup;
  }
  exit_code = EXIT_SUCCESS;

cleanup:
  if (matrix_file != NULL) {
    fclose(matrix_file);
  }
  if (value_file != NULL && fclose(value_file) != 0) {
    exit_code = EXIT_FAILURE;
  }
  if (vector_file != NULL && fclose(vector_file) != 0) {
    exit_code = EXIT_FAILURE;
  }
  free(sa);
  free(sb);
  free(isa);
  free(jsa);
  free(eigenvalues);
  free(residuals);
  free(vectors);
  return exit_code;
}
